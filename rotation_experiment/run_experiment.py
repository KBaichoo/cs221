from PIL import Image
import os
import subprocess

classes = ['left', 'right', 'stay']
cwd = os.getcwd()

different_prediction_rot = 0
same_prediction_rot = 0


def exec_model(filename):
    command = ['python3', '../../test_mnist_classifier.py',
               '--model_file',
               '/Users/kevinbaichoo/Desktop/fall2019/cs221/final_project/mnist_cnn.pt', '--img', filename]
    file_rot = filename.split('.')[0] + '_rot.' + filename.split('.')[1]
    result = subprocess.run(command, stdout=subprocess.PIPE)
    command[-1] = file_rot
    rot_result = subprocess.run(command, stdout=subprocess.PIPE)
    return (result.stdout.decode('utf-8'), rot_result.stdout.decode('utf-8'))


for class_type in classes:
    print('Opening folder:', class_type)
    os.chdir(os.path.join(cwd, class_type))
    for filename in os.listdir('.'):
        if not os.path.isfile(filename) or 'thumb' not in filename or 'rot' in filename:
            print('Skipping %s' % filename)
            continue

        res, rot_res = exec_model(filename)
        if res != rot_res:
            print('filename of diff:', filename)
            print('Result {}\n Rot Result {}'.format(
                res.strip(), rot_res.strip()))
            different_prediction_rot += 1
        else:
            same_prediction_rot += 1
print('Same count: %d, Diff count: %d' %
      (same_prediction_rot, different_prediction_rot))
