import atexit
import torch
import subprocess
import os

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

	image = pipe(prompt).images[0]
	image.save("image.png")
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
