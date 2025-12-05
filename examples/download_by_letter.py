import tartanair as ta
import re
import shutil

tartanair_data_root = '/data/tartanair'

ta.init(tartanair_data_root)
usage = shutil.disk_usage(tartanair_data_root)

print("Free space (GB):", usage.free // (2**30))

available_envs = ta.list_envs() # Returns a dictionary with the available environments. Of form {'local': ['env1', 'env2', ...], 'remote': ['env1', 'env2', ...]}
pattern = r"^[A-B]"
filtered_envs = [env for env in available_envs['remote'] if re.match(pattern, env)]
print("Filtered environments:", filtered_envs)

ta.download(env = filtered_envs, 
              difficulty = ['easy'], 
              modality = ['image', 'depth', 'flow'],  
              camera_name = ['lcam_front', 'lcam_right', 'lcam_back', 'lcam_left', 'lcam_top', 'lcam_bottom'], 
              unzip = False,
              delete_zip = False,
              num_workers = 4)