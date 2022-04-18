# Backend for KURO

## Files

.        
├───app.py         // Flask app main file.   
├───model.py      // Class for classify model and integrated gradients.    
├───layers.py     // Class for custom tensorflow layers. 
├───test.ipynb    // Notebook for test.   
├───data          // Model inputs and feature datasets.    
├───templates   //  Flask templates.  
└───Dockerfile     // Docker build file.

## WS APIs

### Train Model
Train model with train set and params.
- route: train
- namespace: /kuro
- params:
    - train_set: Train set selected.
    - params: Model params.
    - uuid: Model's uuid.
    - weights: Feature weights.

### Feature Attribution
Atrribute input features by selected models.
- route: attribute
- namespace: /kuro
- params:
    - rid: Rid of selected region.
    - models: Array of models' uuid.

### Get Feature
Get selected region's detail features.
- route: region_data
- namespace: /kuro
- params:
    - rid: Rid of selected region.

## Other APIs

### Upload Model
- route: upload
- request:
    - file: Model file saved in localhost.

### Download Model
- route: download
- request:
    - id: Model's uuid.

## Start Service
```
python -m flask run --host=xxx.xxx.xxx.xxx --port=xxx
```