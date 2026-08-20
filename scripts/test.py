import os
import json
from scripts.val import validate

def test(model, test_dataloader, modelname, device, mode_test=False, save_errors=False, save_predictions=False):
    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # Obtenemos los resultados incluyendo la lista de detalles
    test_results = validate(model, test_dataloader, device, use_wandb=False, mode_test=mode_test, save_errors=save_errors)
    
    # Guardamos el resumen en el diccionario
    results[modelname] = {
        "rmse": test_results["rmse"],
        "mse": test_results["mse"],
        "mbe": test_results["mbe"],
        "mae": test_results["mae"],
        "sd": test_results["sd"],
        "errores": test_results["errores"]
    }

    with open(json_dir, "w") as f:
        json.dump(results, f, indent=4)

    # --- results_test.txt ---
    fmt_resumen = "{0:25} | {1:15} | {2:15} | {3:15} | {4:15} | {5:15}\n"
    s_resumen = fmt_resumen.format("Modelo", "MSE", "RMSE (metros)", "MBE (metros)", "MAE (metros)", "SD")
    s_resumen += "-" * 63 + "\n"
    for k, v in results.items():
        s_resumen += fmt_resumen.format(k, f"{v['mse']:.6f}", f"{v['rmse']:.6f}", f"{v['mbe']:.6f}", f"{v['mae']:.6f}", f"{v['sd']:.6f}")
    
    with open(os.path.join(os.getcwd(), 'results_test.txt'), "w") as f:
        f.write(s_resumen)

    # --- predicciones_detalladas.txt ---
    fmt_detallado = "{0:40} | {1:15} | {2:15} | {3:15}\n"
    s_detallado = fmt_detallado.format("Nombre de Imagen", "Real (m)", "Predicho (m)", "Error (m)")
    s_detallado += "-" * 90 + "\n"
    
    for i, item in enumerate(test_results["detalles"]):
        error = test_results["errores"][i]
        s_detallado += fmt_detallado.format(
            item['nombre'], 
            f"{item['real']:.2f}", 
            f"{item['pred']:.2f}", 
            f"{error:.2f}"
        )

    with open(os.path.join(os.getcwd(), 'predicciones_detalladas.txt'), "w") as f:
        f.write(s_detallado)

    print(f"¡Test completado!")
    print(f"-> Resumen en: results_test.txt")
    print(f"-> Lista completa en: predicciones_detalladas.txt")