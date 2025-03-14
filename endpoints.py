from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import uvicorn

app = FastAPI(
    title="Deploy breast cancer model - best model",
    version="0.0.3"
)

# ------------------------------------------------------------
# LOAD THE AI MODEL
# ------------------------------------------------------------
model = joblib.load("model/logistic_regression_model.pkl")


@app.post("/api/v1/predict-breast-cancer", tags=["breast-cancer"])
async def predict(
    radius_mean: float,
    texture_mean: float,
    perimeter_mean: float,
    area_mean: float,
    smoothness_mean: float,
    compactness_mean: float,
    concavity_mean: float,
    concave_points_mean: float,
    symmetry_mean: float,
    fractal_dimension_mean: float,
    radius_se: float,
    texture_se: float,
    perimeter_se: float,
    area_se: float,
    smoothness_se: float,
    compactness_se: float,
    concavity_se: float,
    concave_points_se: float,
    symmetry_se: float,
    fractal_dimension_se: float,
    radius_worst: float,
    texture_worst: float,
    perimeter_worst: float,
    area_worst: float,
    smoothness_worst: float,
    compactness_worst: float,
    concavity_worst: float,
    concave_points_worst: float,
    symmetry_worst: float,
    fractal_dimension_worst: float
):
    dictionary = {
        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'concave points_mean': concave_points_mean,
        'symmetry_mean': symmetry_mean,
        'fractal_dimension_mean': fractal_dimension_mean,
        'radius_se': radius_se,
        'texture_se': texture_se,
        'perimeter_se': perimeter_se,
        'area_se': area_se,
        'smoothness_se': smoothness_se,
        'compactness_se': compactness_se,
        'concavity_se': concavity_se,
        'concave points_se': concave_points_se,
        'symmetry_se': symmetry_se,
        'fractal_dimension_se': fractal_dimension_se,
        'radius_worst': radius_worst,
        'texture_worst': texture_worst,
        'perimeter_worst': perimeter_worst,
        'area_worst': area_worst,
        'smoothness_worst': smoothness_worst,
        'compactness_worst': compactness_worst,
        'concavity_worst': concavity_worst,
        'concave points_worst': concave_points_worst,
        'symmetry_worst': symmetry_worst,
        'fractal_dimension_worst': fractal_dimension_worst
    }

    try:
        df = pd.DataFrame(dictionary, index=[0])
        prediction = model.predict(df)
        prediction = prediction[0]

        if prediction == 1:
            prediction = "Maligno"
        else:
            prediction = "Benigno"

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=prediction
        )
    except Exception as e:
        raise HTTPException(
            detail=str(e),
            status_code=status.HTTP_400_BAD_REQUEST
        )


if __name__ == "__main__":
    uvicorn.run(
        "endpoints:app",
        host="0.0.0.0",
        port=8000
    )
