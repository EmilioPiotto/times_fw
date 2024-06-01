from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
import numpy as np
from .model_service import SimpleNN, SimpleLSTM, LinRegNN
app = FastAPI()

class PredictionRequest(BaseModel):
    data: list
    model_name: str

@app.post("/predict")
async def predict(request: PredictionRequest):
    data = np.array(request.data)
    tensor_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    if request.model_name == "SimpleNN":
        # model_output = model(tensor_data)
        pass
    elif request.model_name == "SimpleLSTM":
        # Handle SimpleLSTM
        pass
    elif request.model_name == "LinRegNN":
        input_size = 10
        output_size = 1
        model = LinRegNN(input_size, output_size)
        checkpoint = torch.load("backend\models\lr01_10.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model_output = model(tensor_data)
    else:
        return {"error": "Model not found"}

    prediction = model_output.squeeze().tolist()
    print("Pred:", prediction)
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
