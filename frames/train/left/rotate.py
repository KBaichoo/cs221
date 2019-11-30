from PIL import Image

with open('./files_imgs', 'r') as f:
    for filename in f:
        filename = filename.strip()
        print('Opening file {}'.format(filename))
        for i in range(30, 360, 30):
            im = Image.open(filename)
            rot_i = im.rotate(i)
            filename_san_ext = filename.split(".jpg")[0]
            rot_i.save(filename + '_rot_{}.jpg'.format(i), 'JPEG')
