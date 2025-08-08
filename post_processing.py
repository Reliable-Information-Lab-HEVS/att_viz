import click
from tqdm import trange
import numpy as np
import codecs, json 
from pathlib import Path


@click.command()
@click.option('--cutoff', default=0.5, help='sigmas')
@click.option('--corrfactor', default=1./3., help='factor to renormalize attention weights (lower is stronger)')
@click.option('--firstignored', default=1, help='first tokens to ignore')
@click.argument('infilepath')
def reprocess_html(infilepath, cutoff, corrfactor, firstignored):
	'''
	Improves weigths visualization interpretability by only visualizing tokens with more than <cutoff> times standard deviations above 
	mean (looking at only the prompt tokens after the first <firstignored> ones), and passing all the weightst to the power of <corrfactor>. Given the problems with shallow initialization, attention for shorter
	prompts tend to focus on the first tokens, the first <firstignored> tokens  in the prompt will have thier attention set to zero
	'''

	infilepath = Path(infilepath)
	outfilepath = infilepath.stem + '_reprocessed' + infilepath.suffix


	with open(infilepath, 'rt') as infile, open(outfilepath, 'wt') as outfile:
	    for line in infile:
	    	if 'const params' in line:
	    		print("payload line characters: %d" % len(line))
	    		
	    		start_idx = line.find('{')
	    		end_idx = line.find('; // HACK')

	    		json_isolate = line[start_idx:end_idx]
	    		payload_dict = json.loads(json_isolate)

	    		base_table = payload_dict['attention']['attn']
	    		corrected_table = []
	    		
	    		for _i in trange(len(base_table)):
	    			corrected_lvl_1 = []
	    		
	    			for _j in trange(len(base_table[_i]), leave=False):

	    				corrected_lvl_2 = []
	    				min_len = 2
	    				for _k in trange(len(base_table[_i][_j]), leave=False):	

	    					if min_len == 2:
		    					min_len = len(base_table[_i][_j][_k])


	    					base_table[_i][_j][_k][:firstignored] = 0
	    					arr = np.array(base_table[_i][_j][_k])
	    					std = np.std(arr[firstignored:min_len])
	    					mean = np.mean(arr[firstignored:min_len])
	    					keep_mask = arr > mean + cutoff * std

	    					keep_mask = keep_mask.tolist()
	    					corrected_lvl_3 = [float(_base_float)**corrfactor if keep_mask[i] else 0 for i, _base_float in enumerate(base_table[_i][_j][_k])]
	    		
	    					corrected_lvl_2.append(corrected_lvl_3)
	    				corrected_lvl_1.append(corrected_lvl_2)
	    			corrected_table.append(corrected_lvl_1)

	    		payload_dict['attention']['attn'] = corrected_table

	    		new_text = json.dumps(payload_dict)
	    		new_line = line[:start_idx] + new_text + line[end_idx:]

	    		outfile.write(new_line)
	    	else:
	    		outfile.write(line)


if __name__ == "__main__":
	reprocess_html()
