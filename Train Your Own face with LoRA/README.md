# Train SDXL On Your Own Face Free 8GB
<p align="center">
  <img width="800" src="Train SDXL On Your Own Face Free (8GB).jpeg" "image">
</p>

##
<br/><br/> 

<font size= "4" >

Learn how to train a custom Stable Diffusion XL (SDXL) LoRA on your own face—completely free and running 100% locally on a consumer 8GB GPU with zero CUDA Out-of-Memory (OOM) crashes.

In this hands-on engineering tutorial, we build a lean, production-grade Python workflow using Hugging Face Diffusers, Accelerate, PEFT, and BitsAndBytes. You will see exactly how to bypass expensive cloud GPU rentals and master memory-efficient fine-tuning on consumer hardware.. <br/>

📌 What You’ll Learn: <br/>

1. High-Performance SDXL LoRA Training: <br/>
   - Pre-encoding CLIP text embeddings to dump both text encoders from VRAM permanently. <br/>
   - Pre-caching VAE latents straight to system RAM to eliminate redundant forward passes. <br/>
   - Applying BitsAndBytes 8-bit AdamW and gradient checkpointing to squeeze training into 8GB VRAM. <br/>
   - Setting up PyTorch CUDA memory allocation flags to avoid fragmentation on Windows / WSL2. <br/>
 <br/>
2. Production Inference & Generation Pipeline: <br/>
   - Loading and scaling your custom face LoRA weights onto SDXL Base. <br/>
   - Utilizing sequential CPU offload and VAE tiling to guarantee stable 8GB inference. <br/>
   - Leveraging DPMSolverMultistepScheduler (SDE-DPMSolver++ with Karras sigmas) for clean 30-step convergence. <br/>
   - Crafting cinematic, editorial, vintage 35mm, and action prompt templates using your custom trigger token. <br/>


🔔 Subscribe for more Python AI, Computer Vision, and Deep Learning tutorials! <br/>


You can find the link for the [tutorial](https://youtu.be/FyYNFWNO_h4) here. <br/>

You can find more Mind-Blowing Computer Vision tutorials in this [playlist](https://www.youtube.com/playlist?list=PLdkryDe59y4bFpkDKfKkPLjfjYWCBdXDu)


Enjoy

Eran
<br/><br/> 

</font>

# Recommended courses and relevant products 
<font size= "4" >

🚀 Want to get started with Computer Vision or take your skills to the next level ? 

If you’re just beginning, I recommend this step-by-step course designed to introduce you to the foundations of Computer Vision - [Complete Computer Vision Bootcamp With PyTorch & TensorFlow](https://trk.udemy.com/9LoE7E) 

If you’re already experienced and looking for more advanced techniques, check out this deep-dive course - [Modern Computer Vision GPT, PyTorch, Keras, OpenCV4](https://trk.udemy.com/EEDyMD)

Before we continue , I actually recommend this [book](https://amzn.to/3STWZ2N) for deep learning based on Tensorflow and Keras : 



</font>

# Connect

<font size= "4" >
If you have any suggestions about papers, feel free to mail me :)

- [☕ Buy me a coffee](https://ko-fi.com/eranfeit)
- [🌐 My Website](https://eranfeit.net)
- [▶️ Youtube.com/@eranfeit](https://www.youtube.com/channel/UCTiWJJhaH6BviSWKLJUM9sg)
- [🐙 Facebookl](https://www.facebook.com/groups/3080601358933585)
- [🖥️ Email](mailto:feitgemel@gmail.com)
- [🐦 Twitter](https://twitter.com/eran_feit )
- [😸 GitHub](https://github.com/feitgemel)
- [📸 Instagram](https://www.instagram.com/eran_feit/)
- [🤝 Fiverr ](https://www.fiverr.com/s/mB3Pbb)
- [📝 Medium ](https://medium.com/@feitgemel)


</font>


