from svg_disturb.disturb import *
import random
from svgpathtools import wsvg, Path
from svgutils.compose import *
from utils.design_sheet_utils import *
from utils.detection_utils import *
import shutil




count = -1 #explain: this variable should be global (outside the resursive parser function) to make it fixed for all the paths on the svg file
def parse(in_node, out_parent, out_svg, ps, M_global, params):
    """
    function: parse a svg file from adobe illustrate into polylines format
    :param in_node: minidom.parse(input file name)
    :type in_node:
    :param out_parent:
    :type out_parent:
    :param out_svg: minidom.Document(output file name)
    :type out_svg:
    :param ps: a list of paths(Path()) in the design: the path is used to get points (polyline format)
    :type ps:
    :return:
    :rtype:
    """
    global count
    if in_node.nodeType != 1:
        return

    # print in_node.nodeName, in_node.nodeType, in_node.nodeValue
    if in_node.nodeName == 'path':
        count = count + 1
        for j in range(len(ps[count])):
            out_node = out_svg.createElement("polyline")
            data = get_points(ps[count][j])

            M_local = getRandomTransform(data['center'], params)
            disturbed_data = disturbPoly(data, np.dot(M_global, M_local), 0 if params['COHERENT'] else params['PER_POINT_NOISE'])
            addOverstroke(disturbed_data, params['OVERSTROKE'], params['UNDERSTROKE'])

            for a in in_node.attributes.keys():
                if a == 'd':
                    out_node.setAttribute("points", ''.join("%0.3f,%0.3f"%(x[0],x[1])+' ' for x in disturbed_data))
                elif a == 'points' or a=="cx" or a=="cy":
                    continue
                else:
                    out_node.setAttribute(a, in_node.getAttribute(a))
            out_parent.appendChild(out_node)
    else:
        out_node = out_svg.createElement(in_node.nodeName)
        for a in in_node.attributes.keys():

            if a=="height":
                out_node.setAttribute(a, str(params['cavans_sizey']))
            elif a=="width":
                out_node.setAttribute(a, str(params['cavans_sizex']))
            elif a == "viewBox":
                out_node.setAttribute(a, "0.0 0.0 %s %s"%(str(params['cavans_sizex']), str(params['cavans_sizey'])))
            else:
                out_node.setAttribute(a, in_node.getAttribute(a))
        out_parent.appendChild(out_node)

    for child in in_node.childNodes:
        parse(child, out_node, out_svg, ps, M_global,  params)


def generate(save_glyph_compose_dir, params_global, params, num_det_imgs, num_glyph_per_image_fix=1000):
    global count
    bounding_boxings = [] #use to check intersection when compositing multi glyphs for detection
    starting_index = 0 #the index to start of new image for detection
    count_det_imgs = 0 # how many detection images have been generated now
    success = 0 #how many images has pass the intersection in the detection
    num_glyph_per_image = num_glyph_per_image_fix


    num_cls_imgs = 10000000

    for i in range(num_cls_imgs):
        print(f"graph number:  {i}")
        count = -1
        selected_elements,  labels = selection()

        save_name = os.path.join(save_glyph_compose_dir,  labels[0] + "_" + labels[1] + "_" + labels[2]  + ".svg")

        # for a, we only need the anchor a (i.e., a_0), as others are pure transformations of the whole design
        #explain: for "fill" attribute, i need to keep paths' original order. see the function for more detail
        a, a_attributes, a_paths = read_element('./thoughts/a.svg')

        if labels[0] == '0':
            a_attributes[0]['fill']='#B2C9EA'
        elif labels[0] == '1':
            a_attributes[0]['fill'] = '#8F9A6F'
        elif labels[0] == '2':
            a_attributes[0]['fill'] = '#799A91'
        elif labels[0] == '3':
            a_attributes[0]['fill'] = '#9A9494'
        elif labels[0] == '4':
            a_attributes[0]['fill'] = '#D9B460'
        elif labels[0] == '5':
            a_attributes[0]['fill'] = '#73D29F'

        b, b_attributes, b_paths = read_element('./thoughts/b.svg')
        b, b_paths = align_b_2_a(b, b_paths, a)
        if labels[1] == '0':
            b_attributes[0]['fill'] = '#4A484E'
        elif labels[1] == '1':
            b_attributes[0]['fill'] = '#855251'

        if labels[2] == '0':
            c, c_attributes, c_paths = read_element('thoughts/c.svg')
            c, c_paths = align_c_2_a(c, c_paths, a)

            ps_combine = Path(*a, *b, *c)
        else:
            ps_combine = Path(*a, *b)


        _, _, _, _, x_center, y_center = get_bounding_box_center(ps_combine)
        #align the center to the center of the canvans

        trans_x = trans_y = 0.0

        a = a.translated(trans_x + 1j*trans_y)
        a_paths = translate_path_list(a_paths, trans_x, trans_y)

        b = b.translated(trans_x + 1j * trans_y)
        b_paths = translate_path_list(b_paths, trans_x, trans_y)
        if labels[2] == '0':
            c = c.translated(trans_x + 1j * trans_y)
            c_paths = translate_path_list(c_paths, trans_x, trans_y)

            # explain: each for one parameter, we only need to change one attribute of one of the paths of this parameter, so here we have seperate [a_paths[0]], [a_paths[1]], ...
            ps = [a_paths, b_paths, c_paths]  # this is a list of indivual paths
            ps_combine = Path(*a, *b, *c)
            wsvg([a, b, c], attributes=[a_attributes[0], b_attributes[0], c_attributes[0]], filename=save_name)
        else:

            ps = [a_paths, b_paths]  # this is a list of indivual paths
            ps_combine = Path(*a, *b)
            wsvg([a, b], attributes=[a_attributes[0], b_attributes[0]], filename=save_name)

        _, _, _, _, x_center, y_center = get_bounding_box_center(ps_combine)

        M_global = getGlobalTransform_Oneside([x_center, y_center], params_global)#explain: the reason of doing this globally is to make sure it is the same among
        # all the paths in the file



        svg_in = minidom.parse(save_name) # read the glyph file
        svg_out = minidom.Document()
        parse(svg_in.documentElement, svg_out, svg_out, ps, M_global, params)

        out_file = open(save_name, 'w')      #write the output file
        out_file.write(svg_out.toprettyxml())
        out_file.close()
        rasterization(save_name)


        # explain: this save name is to be used for later layout composition
        # get the bounding box information regarding to the overall design (big svg canvas) from the final glyph
        final_combined_path, _, file_paths = read_element(save_name)
        detection_img_save_name = os.path.join(save_glyph_compose_dir, str(count_det_imgs) + ".svg")
        label_file = os.path.join(save_glyph_compose_dir, str(count_det_imgs) + ".txt")
        inter = False
        label = '0'

        x_min, x_max, y_min, y_max, _, _ = get_bounding_box_center(final_combined_path)


        bounding_boxing = (x_min, y_min, (x_max - x_min)*0.8, (y_max - y_min)*0.8)
        if i == starting_index:
            Figure(params['cavans_sizex'], params['cavans_sizey'], SVG(save_name)).save(detection_img_save_name)
            success = success + 1
            x_min, x_max, y_min, y_max, _, _ = get_bounding_box_center(final_combined_path)
            bounding_boxings.append(bounding_boxing)
        else:
            for bx in bounding_boxings:
                if is_collision(bounding_boxing, bx):
                    inter = True
                    break
        if not inter:
            Figure(params['cavans_sizex'], params['cavans_sizey'], SVG(save_name), SVG(detection_img_save_name)).save(detection_img_save_name)
            x_min, x_max, y_min, y_max, _, _ = get_bounding_box_center(final_combined_path)
            text = str(count_det_imgs) + " " + str(label) + " " + str(x_min) + " " + str(x_max) + " " + str(y_min) + " " + str(y_max)
            write_file(label_file, text)
            success = success + 1
            bounding_boxings.append(bounding_boxing)

        # clean up
        # remove the glyph image
        os.remove(save_name)  # *.svg
        os.remove(save_name.replace(".svg", ".png"))  # rasterized png

        if success == num_glyph_per_image:
            # clear all_paths for next image
            starting_index = i + 1
            count_det_imgs = count_det_imgs + 1
            rasterization(detection_img_save_name)
            success = 0
            num_glyph_per_image = int(random.uniform(num_glyph_per_image_fix - 3, num_glyph_per_image_fix + 3))
            bounding_boxings = []

        if count_det_imgs == num_det_imgs:
            print("break")
            break



    alphapng2jpg(save_glyph_compose_dir,jpg_quality_mean=60, jpg_quality_var = 35)
    print("rasterizatoin done")
    txt_pattern = os.path.join(save_glyph_compose_dir, '*.txt')
    # Find all .txt files in the source folder
    txt_files = glob.glob(txt_pattern)
    # Define the new folder path
    annotation_folder = os.path.join(save_glyph_compose_dir, "annotations")

    # Create the new folder if it doesn't exist
    os.makedirs(annotation_folder, exist_ok=True)

    # Move each .txt file to the new folder
    for txt_file in txt_files:
        shutil.move(txt_file, annotation_folder)

    jpg_pattern = os.path.join(save_glyph_compose_dir, '*.jpg')
    # Find all .txt files in the source folder
    jpg_files = glob.glob(jpg_pattern)
    # Define the new folder path
    jpg_folder = os.path.join(save_glyph_compose_dir, "images")

    # Create the new folder if it doesn't exist
    os.makedirs(jpg_folder, exist_ok=True)

    # Move each .txt file to the new folder
    for jpg_file in jpg_files:
        shutil.move(jpg_file, jpg_folder)


def main():

    canvans_single_glyphx = 130
    canvans_single_glyphy = 130
    cavans_sizex = 1000
    cavans_sizey = 700

    # todo: check the influence of the global/local disturb, which creates more variations
    params = {}  # the params used in svg_disturb/disturb.py to deform the each stroke locally
    params_global = {}  # the params used in svg_disturb/disturb.py to deform the design globally

    params['MAX_ANGLE'] = 0.05  # angle in radian
    params['MIN_SCALE'] = 0.8  # scale ratio
    params['MAX_SCALE'] = 1.2  # scale ratio
    params['MAX_TRANSLATE'] = 1  # translation distance in svg units
    params['PRESERVE_RATIO'] = False  # bool
    params['PER_POINT_NOISE'] = 0.5  # absolute translation distance in svg units vs COHERENT (i guess it is the same noise everywhere)
    params['OVERSTROKE'] = True  # bool
    params['COHERENT'] = False  # bool
    params['UNDERSTROKE'] = True  # bool

    # the design's global rotation
    params_global['MAX_ANGLE'] = 0.75  # angle in radian
    # the design's global scale
    params_global['MIN_SCALE'] = 1.4  # scale ratio
    params_global['MAX_SCALE'] = 1.8  # scale ratio
    params_global['PRESERVE_RATIO'] = True  # bool
    params_global['PER_POINT_NOISE'] = 0  # absolute translation distance in svg units
    params_global['OVERSTROKE'] = False  # bool
    params_global['COHERENT'] = True  # bool
    params_global['UNDERSTROKE'] = False  # bool
    params['cavans_sizex'] = cavans_sizex
    params['cavans_sizey'] = cavans_sizey
    # the design's global translation of glyph for the layout (the target svg canvas is 800 * 525)
    params_global['MIN_TRANSLATE_X'] = 0  # here 25 is the margin for min x border
    params_global['MAX_TRANSLATE_X'] = cavans_sizex - 0.5 * canvans_single_glyphx  # here 75 is the margin for max x border
    params_global['MIN_TRANSLATE_Y'] = 0
    params_global['MAX_TRANSLATE_Y'] = cavans_sizey - 0.5 * canvans_single_glyphy  # here 75 is the margin for max x border

    save_glyph_compose_dir = '.\\glyph_detection\\thoughts_detection_dataset'
    if not os.path.exists(save_glyph_compose_dir):
        os.mkdir(save_glyph_compose_dir)



    generate(save_glyph_compose_dir, params_global, params, num_det_imgs=100, num_glyph_per_image_fix=45)


if __name__ == "__main__":
    main()