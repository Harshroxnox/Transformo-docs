# Description:
This notebook takes your fully trained 4 bit qlora adapter, 
merges it with Mistral-7b-instruct-v0.3 base model loaded in fp16 
and then converts it into gguf file using llama.cpp<br>
<br>
After that it quantizes the gguf file using Q5_K_M bit quantization
and exports the final 5GB gguf file as output.<br>
<br>
This notebook is made to run on kaggle. It needs 20GB ram and 
50GB of free disk space. It also expects a lora adapter as input.

`Note: This notebook does not need an accelerator(GPU)`