import torch
import torch.nn.functional as F
import numpy as np

#Si quiero separar entre día y noche tengo que poner val_dataloader_day y val_dataloader_night
def validate(model, val_dataloader, device, use_wandb=False, mode_test=False, save_errors=False):

    model.eval() #modelo en modo examen (congela el aprendizaje)

    # Función para evaluar cualquier dataloader (día o noche) y no repetir el código dos veces
    def evaluar_conjunto(dataloader, nombre_conjunto):
        total_mse_loss = 0.0
        total_samples = 0
        detalles_predicciones = []

        # Le decimos a PyTorch que no calcule derivadas para ahorrar memoria
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(device, non_blocking=True)
                labels = batch[1].to(device, non_blocking=True)
               
                nombres = batch[2] if len(batch) > 2 else [f"img_{i}" for i in range(labels.size(0))]

                logits = model(inputs)
                logits = logits.squeeze(dim=1)

                # Pasamos a metros (x10000)
                logits_real = logits * 10000.0
                labels_real = labels * 10000.0

                # Guardamos los resultados individuales
                for i in range(len(nombres)):
                    detalles_predicciones.append({
                        'nombre': nombres[i],
                        'real': labels_real[i].item(),
                        'pred': logits_real[i].item()
                    })

                # Calculamos el error sumado de todo este batch
                mse_batch = F.mse_loss(logits_real, labels_real, reduction='sum').item()

                total_mse_loss += mse_batch
                total_samples += labels.size(0)

        # Calculamos los promedios finales
        if total_samples > 0:
            mse_medio = total_mse_loss / total_samples
            rmse = mse_medio ** 0.5  # La raíz cuadrada nos da el error en metros
        
            errores=np.array([item['pred']- item['real'] for item in detalles_predicciones])
            mbe=np.mean(errores)
            mae=np.mean(np.abs(errores))
            sd=np.std(errores)

            errores=errores.tolist()
        else:
            mse_medio, mbe, rmse, mae, sd= 0.0, 0.0, 0.0, 0.0, 0.0
            errores=[]
            
        # Guardamos los gráficos si usamos Weights & Biases
        if use_wandb:
            import wandb
            wandb.log({f"Val RMSE {nombre_conjunto} (metros)": rmse})

        return mse_medio, rmse, detalles_predicciones, mbe, mae, sd, errores


    #Evalúo todos los datos juntos. Luego si eso puedo separar día/noche

    mse_global, rmse_global, lista_detalles, mbe_global, mae_global, sd_global, errores = evaluar_conjunto(val_dataloader, "Validación")
    # Devolvemos resultados dependiendo de lo que haya pedido el bucle
    if mode_test:
        # Si es el examen final, devolvemos un diccionario con todos los detalles
        print(f"\n[Examen Final] RMSE: {rmse_global:.2f}m")
        return {
            "rmse": rmse_global,
            "mse": mse_global,
            "detalles": lista_detalles,
            "mbe": mbe_global,
            "mae": mae_global,
            "sd": sd_global,
            "errores": errores,
        }
    else:
        # Si estamos en medio del entrenamiento, solo devolvemos el error global
        # para que el 'train.py' sepa si tiene que guardar el modelo
        return rmse_global

