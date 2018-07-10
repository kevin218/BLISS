from argparse import ArgumentParser
from sklearn.externals import joblib

ap = ArgumentParser()
ap.add_argument('-f', '--filename', type=str, required=True, help='The current joblib to convert')
args = vars(ap.parse_args())

group_now_in = joblib.load(args['filename']) # '../../data/group{}_gsc.joblib.save'.format(0)

# Copy over everything
group_now_out = {}
for key in group_now_in.keys():
    group_now_out[key.decode("utf-8")] = group_now_in[key]

verbose = False
if verbose:
    for key in group_now_out.keys():
        try:
            print(key,group_now_out[key].shape)
        except:
            print(key, type(group_now_out[key]))

y,x = 0,1

# Isolate the features that I know I'll need today
group_now_out['ycenters'] = group_now_in[b'centers'][0,:,:,y]
group_now_out['xcenters'] = group_now_in[b'centers'][0,:,:,x]

# Isolate the features I think that I'll need tomorrow
group_now_out['xwidths'] = group_now_in[b'widths'][0,:,:,x]
group_now_out['ywidths'] = group_now_in[b'widths'][0,:,:,y]

group_now_out['offsets'] = group_now_in[b'heights'][1]
group_now_out['heights'] = group_now_in[b'heights'][0]

if verbose:
    for key in group_now_out.keys():
        try:
            print(key,group_now_out[key].shape)
        except:
            print(key, type(group_now_out[key]))

new_filename = args['filename'].replace('_gsc.joblib.save', '_new.joblib.save')
print("Storing new dictionary into {}".format(new_filename))
joblib.dump(group_now_out, new_filename)
