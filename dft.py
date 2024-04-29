import png

def read_img(fn):
    reader = png.Reader(filename= fn)
    width, height, pixels, info = reader.read_flat()
    return width, height, pixels, info

def convert_to_greyscale(fn):
    width, height, pixels, info = read_img(fn)
    for i in pixels:
        print(str(i), end=", ")
    print()
    print(len(pixels))
    saver = png.Writer(greyscale= False, width= width, height= height)
    with open("./greyscale.png", "wb") as img:
        saver.write_array(img, pixels)
    return read_img("./greyscale.png")

#fname = input("Enter the name of the image file: ")
width, height, pixels, info = convert_to_greyscale("./img.png")
print("pixel data: "+ str(len(pixels))+" bytes")
print("width: "+ str(width)+" pixels")
print("height: "+ str(height)+ " pixels")
#for i, pixel in enumerate(pixels):
#    print(i)
