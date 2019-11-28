from PIL import Image

for i in range(8, 2224):
    image_name = 'thumb_y{:06d}'.format(i)
    print('Opening image:', image_name)
    x = Image.open(image_name + '.jpg')
    y = x.crop((135, 70, 575, 435))
    y.save(image_name + '.jpg', 'JPEG')
