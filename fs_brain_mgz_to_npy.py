import glob
import numpy as np 
import nibabel as nib
import argparse

def converter(args):
	
	clinica_CAPS_path = args.clinica_CAPS_path

	# We will gather all brain.mgz's gathered by Freesurfer processing in clinica CAPs format
	# Convert them to numpy's using nibabel, store them in place next to where the brain.mgz's are present

	all_brain_mgzs = glob.glob(clinica_CAPS_path+"/*/*/*/*/*/*/*/brain.mgz")

	print("Total number of brain files ", len(all_brain_mgzs))

	for file in all_brain_mgzs:
		img = nib.load(file)
		data = img.get_fdata()
		data = data[:,:,:]
		npy_convert = np.asarray(data)
		print((npy_convert.shape))
		np.save(file.split('.mgz')[0]+".npy", npy_convert)
		print('Saving ', file.split('.mgz')[0]+".npy")

		if args.rotate:
			print("Saving rotated version as well")
			val = 1
			print(val)

			tmp = npy_convert
			for i in range(256):
				tmp[i] = np.rot90(tmp[i], val)
				tmp[i] = np.rot90(tmp[i], val)
				tmp[i] = np.rot90(tmp[i], val)

			np.save(file.split('.mgz')[0]+"_rotated.npy", tmp)
			print('Saving rotated', file.split('.mgz')[0]+"_rotated.npy")


if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser(description='Convert nifti to numpy')
	arg_parser.add_argument('--clinica_CAPS_path', type=str, default="../clinica_CAPS/", help="Path to clinica fressurfer output CAPS folder")
	arg_parser.add_argument('--rotate', default=False, action='store_true')
	args = arg_parser.parse_args()
	converter(args)	