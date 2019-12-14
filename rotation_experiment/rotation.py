from PIL import Image
import os

classes = ['left', 'right', 'stay']
cwd = os.getcwd()

for class_type in classes:
    print('Opening folder:', class_type)
    os.chdir(os.path.join(cwd, class_type))
    for filename in os.listdir('.'):
        if not os.path.isfile(filename) or 'thumb' not in filename:
            print('Skipping %s' % filename)
            continue

        x = Image.open(filename)
        y = x.rotate(180)
        y.save(filename.split('.')[0] + '_rot' + '.jpg', 'JPEG')
