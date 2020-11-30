import arcpy
from arcpy import env
from arcpy.sa import *

import math
import numpy
import pandas as pd

arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("spatial")
arcpy.CheckOutExtension("GeoStats")

presence = arcpy.GetParameterAsText(0)
elevation_ = arcpy.GetParameterAsText(1)
workspace = arcpy.GetParameterAsText(2)
buffer_distance = float(arcpy.GetParameterAsText(3))
number_of_generated_sets = int(arcpy.GetParameterAsText(4))


env.workspace = workspace
elevation_ = arcpy.Raster(elevation_)
elevation_.save("elev")
arcpy.env.extent = elevation_
######################################################################
arcpy.AddMessage("Dividing presence locations to training and validation sub-datasets for frequency ratio and validation analyses")


def training_validation_presence(point_layer):
    number_of_point = int(arcpy.GetCount_management(point_layer).getOutput(0))
    number_of_val_points = int(0.3 * number_of_point)
    number_of_train_points = number_of_point - number_of_val_points
    arcpy.SubsetFeatures_ga(point_layer, workspace + "\\training.shp", workspace + "\\validation.shp", "70",
                            "PERCENTAGE_OF_INPUT")
    return number_of_train_points, number_of_val_points


number_of_train_val = training_validation_presence(presence)
######################################################################
arcpy.AddMessage("####################################################")
arcpy.AddMessage("Creating and classifying groundwater potential driving factors")


def dem_factors(dem):
    try:
        arcpy.env.parallelProcessingFactor = 0
    except:
        pass

    description = arcpy.Describe(dem)
    pixel_size = description.children[0].meanCellHeight
    elevation = Fill(dem)
    out_flow_direction = FlowDirection(elevation, "FORCE")
    out_flow_direction.save(workspace + "\\flow_dir")
    out_flow_accumulation = FlowAccumulation("flow_dir", "", "FLOAT")
    out_flow_accumulation.save(workspace + "\\flow_acc")
    if pixel_size >= 20:
        out_con2 = Con(arcpy.Raster("flow_acc") > 1000, 1)
    else:
        out_con2 = Con(arcpy.Raster("flow_acc") > 500, 1)
    out_con2.save(workspace + "\\stream")
    StreamToFeature("stream", "flow_dir", "stream_feature", "NO_SIMPLIFY")

    out_euc_distance = EucDistance("stream_feature.shp", "", pixel_size, "")
    river_dist_mask = ExtractByMask(out_euc_distance, dem)
    river_dist_mask.save(workspace + "\\riverdist")
    slope_layer = Slope(dem, "DEGREE", 1)
    slope_layer.save(r"{}\slope".format(workspace))

    m_slope = Con("slope" > 1, "slope", 1)
    slope_radians = m_slope * math.pi / 180.0
    slope_radians = Con(slope_radians > 0, slope_radians, 0.001)
    sca = ((arcpy.Raster("flow_acc") + 1) * pixel_size * pixel_size)
    twi = arcpy.sa.Ln(sca / (arcpy.sa.Tan(slope_radians)))
    twi = Con(twi > 0, twi, 0.01)
    twi.save(r"{}\twi".format(workspace))
    arcpy.Delete_management(workspace + "\\flow_acc")
    return


dem_factors(elevation_)
######################################################################
arcpy.AddMessage("####################################################")
arcpy.AddMessage("Calculating Frequency Ratio (FR)")


def factor_reclassify(factor):
    factor_ras = arcpy.Raster(factor)
    min_factor = float(arcpy.GetRasterProperties_management(factor_ras, "MINIMUM").getOutput(0))
    max_factor = float(arcpy.GetRasterProperties_management(factor_ras, "MAXIMUM").getOutput(0))
    remap = []

    if factor.__contains__("elev"):
        range_factor = max_factor - min_factor
        interval = range_factor / 5
        remap = RemapValue([[min_factor, min_factor + interval, 1],
                            [min_factor + interval, min_factor + 2 * interval, 2],
                            [min_factor + 2 * interval, min_factor + 3 * interval, 3],
                            [min_factor + 3 * interval, min_factor + 4 * interval, 4],
                            [min_factor + 4 * interval, max_factor, 5]])

    elif factor == "twi":
        remap = RemapValue([[min_factor, 8, 0], [8, 12, 1],
                            [12, max_factor, 2]])

    elif factor == "slope":
        try:
            if max_factor >= 40 and min_factor < 10:
                remap = RemapValue([[min_factor, 10, 0], [10, 20, 1],
                                    [20, 30, 2], [30, max_factor, 3]])
            elif 40 > max_factor >= 30 and min_factor > 5:
                remap = RemapValue([[min_factor, 5, 0], [5, 15, 1],
                                    [15, 30, 2], [30, max_factor, 3]])
            elif max_factor < 30 and min_factor < 5:
                remap = RemapValue([[min_factor, 5, 0], [5, 10, 1],
                                    [10, 15, 2], [15, max_factor, 3]])
        except:
            range_factor = max_factor - min_factor
            interval = range_factor / 4
            remap = RemapValue([[min_factor, min_factor + interval, 1],
                                [min_factor + interval, min_factor + 2 * interval, 2],
                                [min_factor + 2 * interval, min_factor + 3 * interval, 3],
                                [min_factor + 3 * interval, min_factor + 4 * interval, 4]])

    elif factor == "riverdist":
        remap = RemapValue([[min_factor, 100, 0], [100, 200, 1],
                            [200, 300, 2], [300, 400, 3], [400, 500, 4], [500, max_factor, 5]])

    layer_rec = Reclassify(factor, "VALUE", remap, "NODATA")
    layer_rec.save("{}_rec".format(str(factor)))
    return


for i in ["slope", "riverdist", "twi", "elev"]:
    factor_reclassify(i)
######################################################################
training = "training.shp"
for i in ["elev_rec", "slope_rec", "riverdist_rec", "twi_rec"]:
    ExtractValuesToPoints(training, i, "{}t".format(i), "INTERPOLATE", "VALUE_ONLY")


def area_of_factor_classes(layer):
    with arcpy.da.SearchCursor(layer, ["COUNT", "VALUE"]) as cursor:
        area_ = []
        for row in cursor:
            area_pixel = float(row[0])
            area_.append(area_pixel)
        return area_


slope_area = area_of_factor_classes("slope_rec")
elevation_area = area_of_factor_classes("elev_rec")
river_area = area_of_factor_classes("riverdist_rec")
twi_area = area_of_factor_classes("twi_rec")


def count_classes_presence(layer_rec, layer_point):
    with arcpy.da.SearchCursor(layer_rec, ["COUNT", "VALUE"]) as cursor2:
        value_ = []
        for row in cursor2:
            value = float(row[1])
            value_.append(value)
    with arcpy.da.SearchCursor(layer_point, ["RASTERVALU"]) as cursor:
        points_ = []
        for z in cursor:
            points = float(z[0])
            points_.append(points)
    points_list = []
    for j in value_:
        count_results = points_.count(j)
        points_list.append(count_results)
    return points_list


slope_tr = count_classes_presence("slope_rec", "slope_rect.shp")
elevation_tr = count_classes_presence("elev_rec", "elev_rect.shp")
river_tr = count_classes_presence("riverdist_rec", "riverdist_rect.shp")
twi_tr = count_classes_presence("twi_rec", "twi_rect.shp")


def fr(points_in_classes, area_of_classes):
    result = []
    area_all = 0
    point_all = 0
    area_of_classes = [float(d) for d in area_of_classes]
    points_in_classes = [float(d) for d in points_in_classes]
    for d in area_of_classes:
        area_all += d
    for k in points_in_classes:
        point_all += k
    for f in range(0, len(area_of_classes)):
        frequency_ratio = (points_in_classes[f] / point_all) / (area_of_classes[f] / area_all)
        result.append(frequency_ratio)
    return result


slope_fr = fr(slope_tr, slope_area)
elevation_fr = fr(elevation_tr, elevation_area)
river_fr = fr(river_tr, river_area)
twi_fr = fr(twi_tr, twi_area)

del [twi_tr, twi_area, river_tr, river_area, elevation_tr, elevation_area, slope_tr, slope_area]

arcpy.CreateFileGDB_management(r"{}".format(workspace), "table.gdb")


def add_fr_to_rec(layer_rec, layer_fr):
    f = str(layer_rec)
    f_ = f[:-4]
    value = layer_fr.__len__()
    value_ = range(0, value)
    list_ = zip(value_, layer_fr)
    list_array = numpy.array(list_)
    struct_array = numpy.core.records.fromarrays(list_array.transpose(), numpy.dtype([('VALUE', 'f8'), ('FR', 'f8')]))
    arcpy.da.NumPyArrayToTable(struct_array, r"{}\table.gdb\{}t".format(workspace, f_))


for i in zip(["slope_rec", "elev_rec", "river_rec", "twi_rec"], [slope_fr, elevation_fr, river_fr, twi_fr]):
    add_fr_to_rec(i[0], i[1])

del [slope_fr, elevation_fr, river_fr, twi_fr]


tables = [r"table.gdb\slopet", r"table.gdb\elevt", r"table.gdb\rivert", r"table.gdb\twit"]
tables2 = ["slope_rec", "elev_rec", "riverdist_rec", "twi_rec"]


def join_field(layer_rec, table):
    arcpy.JoinField_management(layer_rec, "VALUE", table, "VALUE", ["FR"])


for i in zip(tables2, tables):
    join_field(i[0], i[1])
del [tables, tables2]


slope_look = Lookup("slope_rec", "FR")
elev_look = Lookup("elev_rec", "FR")
river_look = Lookup("riverdist_rec", "FR")
twi_look = Lookup("twi_rec", "FR")
slope_look.save(workspace + "\\slope_look")
elev_look.save(workspace + "\\elev_look")
river_look.save(workspace + "\\river_look")
twi_look.save(workspace + "\\twi_look")

for i in [slope_look, elev_look, river_look, twi_look]:
    del i

WSumTableObj = WSTable([["slope_look", "VALUE", 1], ["elev_look", "VALUE", 1],
                        ["river_look", "VALUE", 1], ["twi_look", "VALUE", 1]])
FR_layer = WeightedSum(WSumTableObj)
FR_layer.save(workspace + "\\fr")

del FR_layer, WSumTableObj
######################################################################


def factor_q_rec(raster, fr_percent, fr_acc):
    if type(raster) != arcpy.Raster:
        raster = arcpy.Raster(raster)
    value_minimum = raster.minimum
    value_maximum = raster.maximum
    arr = arcpy.RasterToNumPyArray(raster, nodata_to_value=numpy.nan)
    breakpoints = numpy.nanpercentile(arr, fr_percent)
    remap = RemapValue([[value_minimum, breakpoints, 0],
                        [breakpoints, value_maximum, 1]])

    if str(raster) == "fr":
        layer_rec = Reclassify(raster, "VALUE", remap, "NODATA")
        layer_rec.save("{}_rec".format(str(raster)))

    fr_50_percent = numpy.nanpercentile(arr, fr_acc)
    return fr_50_percent


fr_50 = factor_q_rec("fr", 75, 50)

fr_high = ExtractByAttributes("fr_rec", "VALUE >= 1")
fr_high.save(workspace + "\\fr_high")
arcpy.RasterToPolygon_conversion("fr_high", "fr_high.shp", "NO_SIMPLIFY", "VALUE")

bound = Con(arcpy.Raster("fr") > 0, 1)
bound.save("bound")

arcpy.RasterToPolygon_conversion("bound", "bound2.shp", "NO_SIMPLIFY", "VALUE")
arcpy.Erase_analysis("bound2.shp", "fr_high.shp", "final_layer", '#')

arcpy.Buffer_analysis(training, "buffer.shp", "{} Meters".format(buffer_distance), "FULL", "ROUND", "NONE", "",
                      "PLANAR")
arcpy.Erase_analysis("final_layer.shp", "buffer.shp", "final_layer2", '#')


def density_func(point, layer):
    description = arcpy.Describe(layer)
    pixel_size = description.children[0].meanCellHeight
    arcpy.env.extent = elevation_
    p_dens_out = PointDensity(point, "NONE", pixel_size)
    p_dens_out.save(workspace + "\\density")


density_func("training.shp", elevation_)
density_layer_mask = ExtractByMask("density", "elev")
density_layer_mask.save(workspace + "\\dens_mask")
density_layer_mask2 = arcpy.Raster("dens_mask")

density_90 = factor_q_rec("dens_mask", 90, 90)
dens_remove = Con(arcpy.Raster("dens_mask") > density_90, 1)
dens_remove.save(workspace + "\\dens_remove")
arcpy.RasterToPolygon_conversion("dens_remove", "dens_remove.shp", "NO_SIMPLIFY")
arcpy.Erase_analysis("final_layer2.shp", "dens_remove.shp", "final_boundary", '#')
######################################################################
arcpy.env.randomGenerator = "1 ACM599"
arcpy.AddMessage("####################################################")


def point_generation(final_boundary, n):
    for j in range(0, n):
        if j < n:
            arcpy.AddMessage("Absence training dataset {} was generated.".format(j+1))

            arcpy.CreateRandomPoints_management(workspace, "non_train{}.shp".format(j), final_boundary, "0 0 250 250",
                                                number_of_train_val[0], "0 Meters", "POINT", "0")

            arcpy.AddMessage("Absence validation dataset {} was generated.".format(j+1))
            arcpy.CreateRandomPoints_management(workspace, "non_Val{}.shp".format(j), final_boundary, "0 0 250 250",
                                                number_of_train_val[1], "0 Meters", "POINT", "0")

            arcpy.env.randomGenerator = "{} ACM599".format(j)
    return


point_generation("final_boundary.shp", number_of_generated_sets)

arcpy.AddField_management(training, "CODE", "SHORT", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
arcpy.AddField_management("validation.shp", "CODE", "SHORT", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")


def add_code_field(point_name, number):
    for k in range(0, number):
        try:
            arcpy.AddField_management("{}{}.shp".format(point_name, k), "CODE", "SHORT", "", "", "", "", "NULLABLE",
                                      "NON_REQUIRED", "")
        except:
            pass


add_code_field("non_train", number_of_generated_sets)
add_code_field("non_Val", number_of_generated_sets)


def points_code(layer, value):
    cursor = arcpy.UpdateCursor(layer)
    for row in cursor:
        row.setValue("CODE", value)
        cursor.updateRow(row)
    return


points_code(training, 1)
points_code("validation.shp", 1)


def delete_field(layer):
    fields = arcpy.ListFields(layer)
    delete_list_train = []
    for k in fields:
        if k.name.__contains__("FID"):
            pass
        elif k.name.__contains__("Shape"):
            pass
        elif k.name.__contains__("CODE"):
            pass
        else:
            delete_list_train.append(str(k.name))

    for k in delete_list_train:
        arcpy.DeleteField_management(layer, k)


delete_field(training)
delete_field("validation.shp")

ExtractValuesToPoints(training, "fr", "training2.shp", "INTERPOLATE", "VALUE_ONLY")
ExtractValuesToPoints("validation.shp", "fr", "validation2.shp", "INTERPOLATE", "VALUE_ONLY")


def list_of_files(train_set, validation_set):
    list_files = arcpy.ListFeatureClasses()
    train_list = []
    validation_list = []
    for k in list_files:
        if k.__contains__(validation_set):
            validation_list.append(k)
        elif k.__contains__(train_set):
            train_list.append(k)
    list_final = [train_list, validation_list]
    return list_final


b = list_of_files("non_train", "non_Val")


def extract_value(values, output):
    count = 0
    for k in values:
        ExtractValuesToPoints(k, "fr", "{}{}".format(output, count), "INTERPOLATE", "VALUE_ONLY")
        count += 1


extract_value(b[0], "trset")
extract_value(b[1], "vlset")

a = list_of_files("trset", "vlset")


def merge_presence(layer1, list_of_layers, name):
    count = 0
    for k in list_of_layers:
        arcpy.Merge_management("{}; {}".format(layer1, k), "{}{}.shp".format(name, count))
        count += 1


merge_presence("training2.shp", a[0], "tr_set")
merge_presence("validation2.shp", a[1], "vl_set")
del a
######################################################################


def accuracy_index_q(layer, value_50_percent):
    with arcpy.da.SearchCursor(layer, ["CODE", "RASTERVALU"]) as cursor:
        list_code = []
        list_raster_value = []
        for row in cursor:
            list_code.append(row[0])
            list_raster_value.append(row[1])

        new_list = []
        for k in list_raster_value:
            if k < value_50_percent:
                num = 0
            else:
                num = 1
            new_list.append(num)

        count_true = 0
        count_false = 0
        for k in zip(new_list, list_code):
            if k[0] == k[1]:
                count_true += 1
            else:
                count_false += 1
        true_false = [count_true, count_false]
        acc_index = float(float(true_false[0]) / (float(true_false[0]) + float(true_false[1])))
        return acc_index


c = list_of_files("tr_set", "vl_set")

train_accuracy = []
for i in c[0]:
    acc = accuracy_index_q("{}".format(i), fr_50)
    train_accuracy.append(round(acc, 9))

validation_accuracy = []
for i in c[1]:
    acc = accuracy_index_q("{}".format(i), fr_50)
    validation_accuracy.append(round(acc, 9))

index_train = train_accuracy.index(max(train_accuracy))

index_validation = validation_accuracy.index(max(validation_accuracy))

arcpy.AddMessage("####################################################")
arcpy.AddMessage("Training set {} and Validation set {} were selected with accuracies of {} and {}, respectively"
                 .format(index_train + 1, index_validation + 1, train_accuracy[index_train],
                         validation_accuracy[index_validation]))

arcpy.CreateFolder_management(workspace, "Results")
arcpy.CopyFeatures_management("final_boundary.shp", "Results/final_boundary.shp")
arcpy.CopyFeatures_management("tr_set{}.shp".format(index_train), "Results/tr_set{}.shp".format(index_train+1))
arcpy.CopyFeatures_management("vl_set{}.shp".format(index_validation), "Results/vl_set{}.shp".format(index_validation+1))
arcpy.CopyRaster_management("fr", r"{}\results\fr".format(workspace))

for i in ["Results/tr_set{}.shp".format(index_train+1), "Results/vl_set{}.shp".format(index_validation+1)]:
    arcpy.DeleteField_management(i, ["CID", "RASTERVALU"])
arcpy.DeleteField_management("Results/final_boundary.shp", ["gridcode"])

tr_list = zip(range(1, number_of_generated_sets + 1), train_accuracy)
vl_list = zip(range(1, number_of_generated_sets + 1), validation_accuracy)

tra_list = pd.DataFrame(tr_list)
tra_list.columns = ["training set", "accuracy"]
tr = tra_list.set_index("training set")
val_list = pd.DataFrame(vl_list)
val_list.columns = ["validation set", "accuracy"]
vl = val_list.set_index("validation set")

tr.to_excel(r"{}\results\training.xls".format(workspace))
vl.to_excel(r"{}\results\validation.xls".format(workspace))

list_files_final = arcpy.ListFeatureClasses()
list_files_final2 = arcpy.ListRasters()
list_files_final3 = list_files_final + list_files_final2

for x in list_files_final3:
    arcpy.Delete_management(x)

for i in ["log", "info", "table.gdb"]:
    arcpy.Delete_management(i)
