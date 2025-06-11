from svg_disturb.disturb import *
from svgpathtools import wsvg, Path
from svgutils.compose import *
from utils.design_sheet_utils import *
from utils.detection_utils import *

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


def generate(save_glyph_dir, params_global, params):
    global count
    labels_list = all_label_combinations()
    for i, labels in enumerate(labels_list):
        print(f"graph number:  {i}")
        count = -1
        save_name = os.path.join(save_glyph_dir, labels[0] + "_" + labels[1] + "_" + labels[2]  + ".svg")

        #explain: for "fill" attribute, keep paths' original order. see the function for more detail
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
        b, b_paths = align_b_2_a(b, b_paths, a, shift=0.0)
        if labels[1] == '0':
            b_attributes[0]['fill'] = '#4A484E'
        elif labels[1] == '1':
            b_attributes[0]['fill'] = '#855251'

        if labels[2] == '0':
            c, c_attributes, c_paths = read_element('./thoughts/c.svg')
            c, c_paths = align_c_2_a(c, c_paths, a, shift=0.0)
            ps_combine = Path(*a, *b, *c)
        else:
            ps_combine = Path(*a, *b)


        _, _, _, _, x_center, y_center = get_bounding_box_center(ps_combine)
        #align the center to the center of the canvans
        trans_x = params['cavans_sizex'] / 2.0 - x_center
        trans_y = params['cavans_sizey'] / 2.0 - y_center

        a = a.translated(trans_x + 1j*trans_y)
        a_paths = translate_path_list(a_paths, trans_x, trans_y)

        b = b.translated(trans_x + 1j * trans_y)
        b_paths = translate_path_list(b_paths, trans_x, trans_y)
        if labels[2] == '0':
            c = c.translated(trans_x + 1j * trans_y)
            c_paths = translate_path_list(c_paths, trans_x, trans_y)
            ps = [a_paths, b_paths, c_paths]  # this is a list of indivual paths
            ps_combine = Path(*a, *b, *c)
            wsvg([a, b, c], attributes=[a_attributes[0], b_attributes[0], c_attributes[0]], filename=save_name)
        else:
            ps = [a_paths, b_paths]  # this is a list of indivual paths
            ps_combine = Path(*a, *b)
            wsvg([a, b], attributes=[a_attributes[0], b_attributes[0]], filename=save_name)

        _, _, _, _, x_center, y_center = get_bounding_box_center(ps_combine)

        M_global = getGlobalTransform_Oneside([x_center, y_center], params_global)#explain: the reason of doing this globally is to make sure it is the same among

        svg_in = minidom.parse(save_name) # read the glyph file
        svg_out = minidom.Document()
        parse(svg_in.documentElement, svg_out, svg_out, ps, M_global, params)

        out_file = open(save_name, 'w')      #write the output file
        out_file.write(svg_out.toprettyxml())
        out_file.close()
        rasterization(save_name)


    alphapng2jpg(save_glyph_dir, jpg_quality_mean=60, jpg_quality_var = 35)
    print("rasterizatoin done")



def main():
    canvans_single_glyphx = 130
    canvans_single_glyphy = 130

    # todo: check the influence of the global/local disturb, which creates more variations
    params = {}  # the params used in svg_disturb/disturb.py to deform the each stroke locally
    params_global = {}  # the params used in svg_disturb/disturb.py to deform the design globally
    params['MAX_ANGLE'] = 0.0  # angle in radian
    params['MIN_SCALE'] = 1.0  # scale ratio
    params['MAX_SCALE'] = 1.0  # scale ratio
    params['MAX_TRANSLATE'] = 0  # translation distance in svg units
    params['PRESERVE_RATIO'] = True  # bool
    params['PER_POINT_NOISE'] = 0.0  # absolute translation distance in svg units vs COHERENT (i guess it is the same noise everywhere)
    params['OVERSTROKE'] = False  # bool
    params['COHERENT'] = False  # bool
    params['UNDERSTROKE'] = False  # bool
    params['cavans_sizex'] = canvans_single_glyphx
    params['cavans_sizey'] = canvans_single_glyphy

    # the design's global rotation
    params_global['MAX_ANGLE'] = 0  # angle in radian
    # the design's global scale
    params_global['MIN_SCALE'] = 1.6  # scale ratio
    params_global['MAX_SCALE'] = 1.6  # scale ratio
    params_global['MIN_TRANSLATE_X'] = 0  # here 25 is the margin for min x border
    params_global['MAX_TRANSLATE_X'] = 0  # here 75 is the margin for max x border
    params_global['MIN_TRANSLATE_Y'] = 0
    params_global['MAX_TRANSLATE_Y'] = 0  # here 75 is the margin for max x border
    params_global['PRESERVE_RATIO'] = True  # bool
    params_global['PER_POINT_NOISE'] = 0  # absolute translation distance in svg units

    save_glyph_dir = "..\glyph_rec\\thoughts_rec_dataset\default_glyph"

    if not os.path.exists(save_glyph_dir):
        os.makedirs(save_glyph_dir)

    generate(save_glyph_dir, params_global, params)


if __name__ == "__main__":
    main()