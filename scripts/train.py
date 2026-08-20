import torch
import torch.nn as nn
import os
from tqdm import tqdm
import wandb
import gc
from scripts.val import validate # ¡Recuerda que luego tendremos que adaptar esta función también!
from utils.utils import freeze_backbone_layers, unfreezing_scheduler

#CARPETA CHECKPOINTS
#ruta_carpeta = '/content/drive/MyDrive/TFG/checkpoints'
ruta_carpeta='./checkpoints'
os.makedirs(ruta_carpeta, exist_ok=True)


def train_regression(model, optimizer, scheduler, train_dataloader, val_dataloader, criterion, device, use_wandb=True, epochs=1000, 
            verbose = 20, modelname= 'model', use_amp= True, out_path='results/', patience=100, accum_steps=1, freeze_steps=5, freeze=False, warmup=False, refine_flag=False):

    steps = 0
    # En regresión buscamos el error más bajo, por lo que inicializamos el mejor error en infinito
    best_val_error = float('inf') 
    early_stopping_counter = 0

    # Creamos la carpeta de resultados si no existe
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    scaler = torch.amp.GradScaler(enabled=use_amp) #entrena a la red más rápido

    # BUCLE PRINCIPAL DE ÉPOCAS. Una época es cuando la red ha visto todas las fotos una vez

    start_epoch=0
    ruta_ultimo_checkpoint = os.path.join(ruta_carpeta, "ultimo_checkpoint.pt")

    if os.path.exists(ruta_ultimo_checkpoint):
        print(f"Encontrado checkpoint previo. Reanudando...")
        checkpoint_guardado = torch.load(ruta_ultimo_checkpoint)
        
        # Restauramos los pesos
        model.load_state_dict(checkpoint_guardado['estado_modelo'])
        optimizer.load_state_dict(checkpoint_guardado['estado_optimizador'])
        start_epoch = checkpoint_guardado['epoca'] + 1
    else:
        print("No hay guardados previos. Empezando desde cero...")
    
    for epoch in range(start_epoch, epochs):

        if refine_flag:
            unfreezing_scheduler(model, epoch, epochs)

        if freeze == True and refine_flag == False:
            if epoch < freeze_steps:
                freeze_backbone_layers(model, freeze=True)
            else:
                freeze_backbone_layers(model, freeze=False)

        model.train() # Ponemos el modelo en modo entrenamiento
        running_loss = 0.0 # Guardará el MSE total
        total_samples = 0

        # BUCLE DE BATCHES (Lotes de imágenes)
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = batch[0].to(device, non_blocking=True) #imágenes
            labels = batch[1].to(device, non_blocking=True) #alturas de las imagenes

            try:
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    # 1. La red hace la predicción
                    logits = model(inputs)
                    
                    # 2. Ajustamos dimensiones: de [batch_size, 1] a [batch_size]
                    logits = logits.squeeze(dim=1) 
                    
                    # 3. Calculamos el Error Cuadrático Medio (MSE)
                    loss = criterion(logits, labels)
                    loss = loss / accum_steps #accum_steps es para que el ordenador trate imágenes de menos en menos, por ejemplo grupos de 4.
                    #el error medio lo va a dividir entre este número de steps para asemejar a que has metido todas las fotos de golpe

                if torch.isnan(loss).any():
                    print("Warning: NaN loss encountered. Skipping batch.")
                    continue

                # 4. Aprendemos del error (Backward pass)
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (steps + 1) % accum_steps == 0:
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                steps += 1

                # Extraemos el MSE puro para las estadísticas. Aquí podemos hacer otros cálculos que necesitemos luego para predicciones
                mse = loss.item() * accum_steps 

                # Logueamos en Weights & Biases (opcional)
                if use_wandb:
                    wandb.log({"Train MSE Loss": mse})
                    wandb.log({"Train RMSE (meters)": (mse ** 0.5)*10000}) # La raíz cuadrada nos da el error en metros

                running_loss += mse * labels.size(0)
                total_samples += labels.size(0)

            except Exception as e:
                print(f"Error during training step: {e}")
                continue
                
        # Actualizamos el learning rate
        if warmup == False:
            scheduler.step()
        else:
            scheduler.step(epoch)

        # ESTADÍSTICAS AL FINAL DE LA ÉPOCA
        epoch_mse = running_loss / total_samples if total_samples > 0 else float('nan')
        epoch_rmse = (epoch_mse ** 0.5)*10000 if epoch_mse > 0 else float('nan')

        print(f"\nEpoch {epoch}: Train MSE = {epoch_mse:.4f} | Train RMSE = {epoch_rmse:.2f} meters")

        # FASE DE VALIDACIÓN (Examen)
        if epoch % verbose == 0 or epoch == epochs - 1:

            # validate() debe devolver el error RMSE en metros
            val_error = validate(model, val_dataloader, device, use_wandb, mode_test=False, save_errors=False)

            if refine_flag:
                torch.save(model.state_dict(), os.path.join(out_path, f"{modelname}_{epoch}.pt"))
            else:
                # Si el error es menor que nuestro mejor registro histórico, guardamos el modelo
                if val_error < best_val_error:
                    best_val_error = val_error
                    early_stopping_counter = 0 
                    print(f"--> ¡Mejora detectada! Nuevo mejor error de validación: {best_val_error:.2f} metros. Guardando modelo...")
                    torch.save(model.state_dict(), os.path.join(out_path, f"{modelname}.pt"))
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered. El modelo ha dejado de mejorar.")
                        break


        print(f"Época {epoch} terminada.")
        ruta_archivo=f"{ruta_carpeta}/checkpoint_epoca_{epoch}.pt"
        torch.save
        torch.save({
            'epoca': epoch,
            'estado_modelo': model.state_dict(),
            'estado_optimizador': optimizer.state_dict(),
            'perdida':epoch_rmse # Opcional: guardar el loss actual
        }, ruta_ultimo_checkpoint)
        
        
        gc.collect()
