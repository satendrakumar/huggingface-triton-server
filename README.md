

### Build the conda env 
     $ cd model_repository/sentiment
     $ sh build-env.sh

### Start docker container:
       $ docker run -d --shm-size=10G -p8000:8000 -p8001:8001 -p8002:8002 --gpus device=0  -v $PWD/model_repository:/mnt/model_repository nvcr.io/nvidia/tritonserver:23.06-py3 tritonserver --model-repository=/mnt/model_repository --log-verbose=1
       

#### Hello example from Triton:
       $ curl --location --request POST 'http://localhost:8000/v2/models/hello/infer' \
         --header 'Content-Type: application/json' \
         --data-raw '{
            "inputs":[
            {    
             "name": "text",
             "shape": [1],
             "datatype": "BYTES",
             "data":  ["Hello world"]
            }
           ]
         }'  
         Response:{"model_name":"hello","model_version":"1","outputs":[{"name":"text","datatype":"BYTES","shape":[1],"data":["Hello from Triton Hello world"]}]}
#### HuggingFace model Inference using CURL:
         $ curl --location --request POST 'http://localhost:8000/v2/models/sentiment/infer' \
             --header 'Content-Type: application/json' \
             --data-raw '{
                "inputs":[
                {    
                 "name": "text",
                 "shape": [1],
                 "datatype": "BYTES",
                 "data":  ["I really enjoyed this"]
                }
               ]
             }'  

     Response:
         {"model_name":"sentiment","model_version":"1","outputs":[{"name":"sentiment","datatype":"BYTES","shape":[1],"data":["positive"]}]}