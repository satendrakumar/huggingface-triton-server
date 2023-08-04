## build conda env for triton server
conda create -k -y -n hf-sentiment python=3.10
conda activate hf-sentiment
pip install numpy conda-pack
pip install torch==1.13.1
pip install transformers==4.21.3
conda install -c conda-forge gcc=12.1.0 # optional if you get issue "nvidia triton version `GLIBCXX_3.4.30' not found"
conda pack -o hf-sentiment.tar.gz

