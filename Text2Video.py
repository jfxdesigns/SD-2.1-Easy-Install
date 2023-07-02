import atexit
import torch
import subprocess
import os
os.system("winget install --id Git.Git -e --source winget")
os.system("winget install --id=Python.Python.3.10  -e")
os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
os.system("git lfs install")
os.system("git clone https://huggingface.co/damo-vilab/text-to-video-ms-1.7b")
os.system("pip install diffusers transformers accelerate torch")
os.system("cls")

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
	from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
	from diffusers.utils import export_to_video
	pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
	pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
	pipe.enable_model_cpu_offload()
	# prompts for an input to text to video
	os.system("cls")
	print("													  ")		
	print("													  ")
	print("													  ")		
	print("													  ")
	print("													  ")		
	print("													  ")
	print("---------------------------------------------------------------------------------------------------")
	print("|              Custom script by https://www.linkedin.com/in/joshua-roberts-bb167a211/             |")
	print("---------------------------------------------------------------------------------------------------")
	print("|											          ")
	print("|												  ")
	print("---------------------------------------------------------------------------------------------------")
	print("|                         To generate higher inference steps or frame count,                      |")
	print("|                   edit the num_inference_steps and num_frames parameters in run.py              |")
	print("---------------------------------------------------------------------------------------------------")
	print("|												  |")
	print("|											          |")
	print("---------------------------------------------------------------------------------------------------")
	print("|                               What would you like to generate?                                  |")
	print("---------------------------------------------------------------------------------------------------")
	promptvariable = input()
	prompt = promptvariable
	# if video does not render, lower num_frames or inference_steps until video renders
	video_frames = pipe(prompt, num_inference_steps=25, num_frames=50).frames
	video_path = export_to_video(video_frames)
	print("'Video rendered with' + num_inference_steps + 'inference steps, and ' + num_frames 'frames.'")
	# print output
	print("---------------------------------------------------------------------------------------------------")
	print("| Your video has been saved at" + video_path +"|")
	print("---------------------------------------------------------------------------------------------------")
	torch.cuda.empty_cache()
	os.system("start " + video_path)
	open(video_path)
	print("---------------------------------------------------------------------------------------------------")
	print("|                       Would you like to generate another video? (y/n)                           |")
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