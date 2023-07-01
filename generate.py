import atexit
import torch
import subprocess
import os
import string
import random

def exitfunction():
	print("---------------------------------------------------------------------------------------------------")	
	print("|                                             cleaning up                                         |")
	print("---------------------------------------------------------------------------------------------------")
	torch.cuda.empty_cache()

atexit.register(exitfunction)

def main():
	print("---------------------------------------------------------------------------------------------------")
	print("|					    ~Loading AI~		  	                 |")
	print("---------------------------------------------------------------------------------------------------")
	from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
	import torch
	model_id = "stabilityai/stable-diffusion-2-1-base"
	scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
	pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
	pipe = pipe.to("cuda")
	pipe.enable_vae_tiling()
	pipe.enable_attention_slicing("max")
	pipe.enable_xformers_memory_efficient_attention(attention_op=None)
	pipe.unet.to(memory_format=torch.channels_last)
	pipe.enable_sequential_cpu_offload()
	# prompts for an input to text to video
	print("													  ")		
	print("													  ")		
	print("													  ")
	print("---------------------------------------------------------------------------------------------------")
	print("|              Custom script by https://www.linkedin.com/in/joshua-roberts-bb167a211/             |")
	print("---------------------------------------------------------------------------------------------------")
	print("---------------------------------------------------------------------------------------------------")
	print("|                               What would you like to generate?                                  |")
	print("---------------------------------------------------------------------------------------------------")
	promptvariable = input()
	prompt = promptvariable
	random_name_length = 7
	res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=random_name_length))
	image = pipe(prompt).images[0]
	os.chdir('outputs')
	image.save(res + ".png")
	torch.cuda.empty_cache()
	print("---------------------------------------------------------------------------------------------------")
	print("|                       Would you like to generate another image ?(y/n)                           |")
	print("---------------------------------------------------------------------------------------------------")
	n = "n"
	y = "y"
	generate_more = input()
	if generate_more == y:
		main()
	if generate_more == n:
		exit()
	else:
		exit()
	
main()
