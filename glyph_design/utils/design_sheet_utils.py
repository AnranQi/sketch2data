import glob
import sys
sys.path.append("..")
from svgpathtools import svg2paths2,wsvg, Path, polygon, Document, disvg, paths2svg, real, imag, Line, CubicBezier, QuadraticBezier, Arc
from svg_disturb.disturb import *
from xml.dom import minidom
from cairosvg import svg2png
from PIL import Image
from shapely.geometry import Polygon, LineString, Point
import numpy as np

def align_b_2_a(b, b_paths, a, shift=1.0):

    movable = b
    movable_paths = b_paths
    anchor = a
    movable_x = "center_x"
    movable_y = "center_y"
    anchor_x = "center_x"
    anchor_y = "center_y"

    movable_x_after = random.uniform(-shift, shift)
    movable_y_after = random.uniform(-shift, shift)

    # alignment function
    b, new_b_paths = movable_2_anchor(movable, movable_paths, movable_x, movable_y, anchor, anchor_x, anchor_y, movable_x_after, movable_y_after)
    return b, new_b_paths

def align_c_2_a(c, c_paths, b, shift=1.0):

    movable = c
    movable_paths = c_paths
    anchor = b
    movable_x = "center_x"
    movable_y = "center_y"
    anchor_x = "center_x"
    anchor_y = "center_y"

    movable_x_after = random.uniform(-shift, shift)
    movable_y_after = random.uniform(-shift, shift)

    # alignment function
    c, new_c_paths = movable_2_anchor(movable, movable_paths, movable_x, movable_y, anchor, anchor_x, anchor_y, movable_x_after, movable_y_after)
    return c, new_c_paths

def selection():
    """
    function: sample a list of values for all the parameters
    :return: l (e.g., ['a_0', 'b_0', 'c_0', 'g_2']): the list of the selected parameters and its values
    label (int): the category label of the glyph composed by the select values
    :rtype:
    """
    l = []
    labels = [] #labels for each parameter


    a = ["0","1","2", "3","4","5"]
    a_select = random.choice(a)
    l.append("a_" + a_select)
    labels.append(a_select)

    b = ["0","1"]
    b_select = random.choice(b)
    l.append("b_" + b_select)
    labels.append(b_select)

    c = ["0", "1"]
    c_select = random.choice(c)
    l.append("c_" + c_select)
    labels.append(c_select)

    return l, labels

def all_label_combinations():
    """
    Generate all possible combinations of parameter values (as labels).

    :return: A list of label combinations. Each label is a list like ['0', '1', '0'].
    """
    a = ["0", "1", "2", "3", "4", "5"]
    b = ["0", "1"]
    c = ["0", "1"]

    all_combos = list(itertools.product(a, b, c))
    return [list(combo) for combo in all_combos]


def rasterization(file):
    """
    function: rasterize a svg file into image
    :param file: svg file path + name
    :type file:
    """
    svg_in = minidom.parse(file)
    svg2png(bytestring=svg_in.toprettyxml(), write_to=file.replace(".svg", ".png"))
    print("done")

# Function to add Gaussian noise to an image
def add_gaussian_noise(image_array, mean=0, std=5):
    std = random.uniform(std-3, std+3)
    gaussian_noise = np.random.normal(mean, std, image_array.shape)
    noisy_image_array = image_array + gaussian_noise
    noisy_image_array = np.clip(noisy_image_array, 0, 255)  # Ensure values are within valid range
    return noisy_image_array.astype('uint8')

def alphapng2jpg(dir, jpg_quality_mean=75, jpg_quality_var = 20):
    files = glob.glob(dir + "/*.png")
    for file in files:

        png = Image.open(file).convert('RGBA')

        png.load()  # required for png.split()

        background = Image.new("RGB", png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
        # Convert the image to a numpy array
        image_array = np.array(background)
        # Add Gaussian noise to the image
        noisy_image_array = add_gaussian_noise(image_array)
        # Convert back to PIL image
        noisy_image = Image.fromarray(noisy_image_array)
        # Save the noisy image as JPEG
        quality = int(random.uniform(jpg_quality_mean - jpg_quality_var, jpg_quality_mean + jpg_quality_var))  # Default quality 75 within range(55,95)
        jpeg_path = file.replace(".png", ".jpg")
        noisy_image.save(jpeg_path, 'JPEG', quality=quality)



def read_element(file):
    """
    function: read the svg file of a parameter value (can be multiple svg paths, will combine into one Path)
    :param file: file name
    :type file: string e.g., './week36/a.0.svg'
    :return
    combined_path: one Path() with all the paths in the file as segments
    svg_attributes: attributes of the all paths in the file
    paths: a list of Path() objects
    """
    #fixme: for all the paths in the file, here i assume they share the same attribute, which is usually true, but need to be careful!
    paths, attributes, svg_attributes = svg2paths2(file)


    for i in range(len(paths)):
        if i==0:
            combined_path = Path(*paths[i])
        else:
            combined_path = Path(*combined_path, *paths[i])

    return combined_path, attributes, paths

def get_bounding_box_center(path):
    """
    function: get th bounding box of a path
    :param path: can be Path or (list of paths)
    :type path:
    :return: xmin, xmax, ymin, ymax, xcenter, ycenter
    :rtype: float
    """
    xmin, xmax, ymin, ymax = paths2svg.big_bounding_box(path)
    return xmin, xmax, ymin, ymax, (xmax-xmin)/2.0 +xmin, (ymax-ymin)/2.0+ymin


def translate_path_list(list_paths,trans_x, trans_y):
    """
    function: translate a list of paths (trans_x, trans_y)
    :param list_paths:
    :type list_paths:
    :param trans_x:
    :type trans_x:
    :param trans_y:
    :type trans_y:
    :return: list_paths
    :rtype: a list of paths
    """
    for i, _ in enumerate(list_paths):
        list_paths[i] = list_paths[i].translated(trans_x + 1j * trans_y)
    return list_paths
def get_points(parameter_curve):
    """
    function: get the points from parametric curve (parameter)
    :param parameter_curve: the parametric parameter curve
    :type parameter_curve:  path

    :return: poly
    poly["points"]: a list of points of the parametric source curve
    poly["center"]: the center of the curve (array)
    :rtype: dict
    """
    length = int(parameter_curve.length())
    numbers = np.linspace(0, 1, length)
    print(length)
    #numbers = np.linspace(0, 1, 50)

    poly = {}
    point_list = []
    for t in numbers:
        # parameter
        p = parameter_curve.point(t)
        px, py = real(p), imag(p)
        point_list.append([px, py])
    poly['points'] = point_list
    center = [0, 0]
    for x in point_list: center = np.sum([center, x], axis=0)
    poly['center'] = np.array(center) / len(point_list)
    return poly


def movable_2_anchor(movable, movable_paths, movable_x, movable_y, anchor, anchor_x, anchor_y, movable_x_after = 0.0, movable_y_after=0.0):
    """
    this function is used to attach the movable to the anchor, i.e., transform the movable to the position of the anchor
    note: the coordinate is based on AI coordinate

    @param movable:  the combined path of movable
    @type movable:
    @param movable_paths: a list of paths of the movable
    @type movable_paths:
    @param movable_x: the attach point of x on the movable, can be "center_x", "min_x", "max_x", or a float number from the min_x to max_x
    @type movable_x:
    @param movable_y: the attach point of y on the movable, values same as the movable_x
    @type movable_y:
    @param anchor: the combined path of anchor
    @type anchor:
    @param anchor_x: the attach point of x on the anchor, values same as the anchor_x
    @type anchor_x:
    @param anchor_y: the attach point of y on the anchor, values same as the anchor_x
    @type anchor_y:
    @param movable_x_after: this is used for align (not attach function), after attachment, how many the moveable should move in x direction
    @type float
    @param movable_y_after: this is used for align (not attach function), after attachment, how many the moveable should move in y direction
    @type float


    """
    movable_xmin, movable_xmax, movable_ymin, movable_ymax, movable_xcenter, movable_ycenter = get_bounding_box_center(movable)
    anchor_xmin, anchor_xmax, anchor_ymin, anchor_ymax, anchor_xcenter, anchor_ycenter = get_bounding_box_center(anchor)
    #x for movable
    if movable_x == "center_x":
        movable_x_coor = movable_xcenter
    elif movable_x == "min_x":
        movable_x_coor = movable_xmin
    elif movable_x == "max_x":
        movable_x_coor = movable_xmax
    else:
        movable_x_coor = float(movable_x) * (movable_xmax - movable_xmin) + movable_xmin
    #y for movable
    if movable_y == "center_y":
        movable_y_coor = movable_ycenter
    elif movable_y == "min_y":
        movable_y_coor = movable_ymin
    elif movable_y == "max_y":
        movable_y_coor = movable_ymax
    else:
        movable_y_coor = float(movable_y) * (movable_ymax - movable_ymin) + movable_ymin

    #x for anchor
    if anchor_x == "center_x":
        anchor_x_coor = anchor_xcenter
    elif anchor_x == "min_x":
        anchor_x_coor = anchor_xmin
    elif anchor_x == "max_x":
        anchor_x_coor = anchor_xmax
    else:
        anchor_x_coor = float(anchor_x) * (anchor_xmax - anchor_xmin) + anchor_xmin
    # y for anchor
    if anchor_y == "center_y":
        anchor_y_coor = anchor_ycenter
    elif anchor_y == "min_y":
        anchor_y_coor = anchor_ymin
    elif anchor_y == "max_y":
        anchor_y_coor = anchor_ymax
    else:
        anchor_y_coor = float(anchor_y) * (anchor_ymax - anchor_ymin) + anchor_ymin


    movable_2_anchor_x  = anchor_x_coor -movable_x_coor + movable_x_after
    movable_2_anchor_y  = anchor_y_coor -movable_y_coor + movable_y_after

    new_movable_paths = []
    movable = movable.translated(movable_2_anchor_x + 1j * movable_2_anchor_y)
    for m_path in movable_paths:
        m_path = m_path.translated(movable_2_anchor_x + 1j * movable_2_anchor_y)
        new_movable_paths.append(m_path)
    return movable, new_movable_paths


####texture related#####
def path_to_polygon(path, scale=1.0):
    """Convert a path to a Shapely Polygon."""
    points = []

    for segment in path:
        if segment:
            points.append((segment.start.real * scale, segment.start.imag * scale))
            #explain: 0.01 is to jitter the point a bot, to avoid generate two same points which cause problem in shapely "cover" function
            points.append((segment.end.real * scale + 0.01, segment.end.imag * scale +0.01))
    return Polygon(points)


def check_point_containment(point, shape_path):
    """Check if the one point is inside the shape path.
    return inside or not
    note shape_path is one single path
    @param point:  the point that needs to be checked
    @type point: [x,y]
    @param shape_path: one shape Path() explain: we currently only support shape with one path()
    @type shape_path:
    @return:
    @rtype: """
    shape_polygon = path_to_polygon(shape_path)
    shapely_point = Point(point)
    return shape_polygon.contains(shapely_point)
