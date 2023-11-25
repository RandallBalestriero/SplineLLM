import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
from transformers import AutoTokenizer
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True)


def get_tokens(prompt):
    input_ids = tokenizer(prompt)['input_ids']
    input_text = []
    for input in input_ids:
        input_text.append(tokenizer.decode(input))#, skip_special_tokens=True)
    return input_text    

data_path = "/home/ubuntu/polytope_hyperplane_side/toxic_pile/imshow_statistics.csv"
data = pd.read_csv(data_path)

X = []
X_prompt = []
for index, row in data.iterrows():
    result_list = ast.literal_eval(row['stats'])
    X.append(np.array(result_list))
    X_prompt.append(row['prompt'])


#plt.style.use('ggplot')
plt.rcParams['font.family'] = 'DejaVu Sans Mono'

print(len(X))
for k in range(len(X)):
    fig, ax = plt.subplots(figsize=(120,8))
    im = ax.imshow(X[k][ :, 0], aspect='auto', cmap='plasma')
    tokens = get_tokens(X_prompt[k])
    plt.xticks(np.arange(X[k].shape[-1]), tokens[:1024], rotation=90, fontsize=8)
    #cbar = plt.colorbar()
    #cbar.ax.yaxis.set_ticks_position('right') 
    plt.grid(False)
    axins = inset_axes(ax,
                        width="1%",  
                        height="100%",
                        loc='right',
                        borderpad=-5
                    )
    fig.colorbar(im, cax=axins, orientation="vertical")    
    #plt.tight_layout()
    plt.savefig(f"/home/ubuntu/polytope_plots/imshow_local_sign_sample_{k}.png", dpi=100)
   

            
            
        # X_ = []
        # import ast 
        # X_prompt = []
        # for data_name in data_names:
        #     for index, row in data[data_name].iterrows():
        #         result_list = ast.literal_eval(row['stats'])

        #         X_.append(np.array(result_list)[:, 2].reshape(-1))
        #         X_prompt.append(row['prompt'])
            