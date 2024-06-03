import laspy
import pandas as pd
from bs4 import BeautifulSoup
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def lazReadWrite_building(laz_file):
    # odczyt pliku laz - chmura punktów
    las_file = laspy.read(laz_file)
    bpoints = las_file.points[las_file.classification == 6]
    x = bpoints.X / 100
    y = bpoints.Y / 100
    z = bpoints.Z / 100
    min_x, max_x, min_y, max_y = x.min(), x.max(), y.min(), y.max()
    data = {'X': x, 'Y': y, 'Z': z}
    df = pd.DataFrame(data)
    print('bbox:', min_x, max_x, min_y, max_y)
    return df, [min_x, max_x, min_y, max_y]

def gmlReadWrite(gml_dictionary_path, cloud_bbox):
    # Read the gml file
    i = 0
    roof_data = {}
    for entry in os.listdir(gml_dictionary_path):
        if os.path.isfile(os.path.join(gml_dictionary_path, entry)):
            with open(f'{gml_dictionary_path}/{entry}', 'r') as f:
                xml_file = f.read()
                soup = BeautifulSoup(xml_file, 'xml')
                # Get the bounding box of the building
                # building_bbox = soup.find('boundedBy')
                env = soup.find('Envelope')
                if env is None:
                    print(f'{entry} has no envelope')
                    continue
                building_bbox_lower = env.find('lowerCorner')
                building_bbox_upper = env.find('upperCorner')
                building_bbox_lower = building_bbox_lower.text.split(' ')
                building_bbox_upper = building_bbox_upper.text.split(' ')
                building_bbox = [building_bbox_lower[0], building_bbox_upper[0], building_bbox_lower[1], building_bbox_upper[1]]
                # check if building bbox intersects cloud bbox 
                if float(building_bbox[0]) < cloud_bbox[1] and float(building_bbox[1]) > cloud_bbox[0] and float(building_bbox[2]) < cloud_bbox[3] and float(building_bbox[3]) > cloud_bbox[2]:
                    print(f'Processing {entry}...')
                    roofs = soup.find_all('RoofSurface')
                    for roof in roofs:
                        RoofPoints = []
                        skipRoof = False
                        points = roof.find_all('pos')
                        for point in points:
                            # print(point.text)
                            x = float(point.text.split(' ')[0])
                            y = float(point.text.split(' ')[1])
                            z = float(point.text.split(' ')[2])
                            if x < cloud_bbox[0] or x > cloud_bbox[1] or y < cloud_bbox[2] or y > cloud_bbox[3]:
                                skipRoof = True
                                break
                            RoofPoints.append([x, y, z])
                        if skipRoof:
                            continue
                        i += 1
                        roof_data[roof['gml:id']] = RoofPoints
                else:
                    print(f'{entry} not in cloud bbox')
                    continue
                    
    print('Number of roofs:', i)
    return roof_data
                        
def roofPointPlanes(roofPoints_dict):
    planes_dict = {}
    for roof in roofPoints_dict:
        # get x and y coordinates
        xAll = [float(point[0]) for point in roofPoints_dict[roof]]
        yAll = [float(point[1]) for point in roofPoints_dict[roof]]
        zAll = [float(point[2]) for point in roofPoints_dict[roof]]
        xArray = np.array(xAll)
        yArray = np.array(yAll)
        zArray = np.array(zAll)
        A = np.c_[xArray, yArray, np.ones(xArray.shape)]
        b = zArray
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        bbox = [min(xAll), max(xAll), min(yAll), max(yAll)]
        planes_dict[roof] = x, bbox
    return planes_dict

def distance(planes_dict, cloudPoints):
    xCloud = cloudPoints['X']
    yCloud = cloudPoints['Y']
    zCloud = cloudPoints['Z']
    distance_array = []
    i = 0
    for roof in planes_dict:
        a, b, d = planes_dict[roof][0]
        c = -1
        x_min, x_max, y_min, y_max = planes_dict[roof][1][0], planes_dict[roof][1][1], planes_dict[roof][1][2], planes_dict[roof][1][3] 
        mask = (x_min <= xCloud) & (xCloud <= x_max) & (y_min <= yCloud) & (yCloud <= y_max)
        distance = (a * xCloud[mask] + b * yCloud[mask] + c * zCloud[mask] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
        gDistance = distance[(distance > -1) & (distance < 1)]
        avDist = np.mean(gDistance)
        distance_array.append((roof, avDist)) 
        print('roof', i, 'done') 
        i += 1 
    return distance_array

def draw_planes(planes, distArray):
    print('Drawing planes...')
    dist_array = {dist[0]: abs(dist[1]) for dist in distArray}
    min_dist, max_dist = min(dist_array.values()), max(dist_array.values())
    fig, ax = plt.subplots(figsize=(10, 10))

    for i, (key, plane) in enumerate(planes.items()):
        x = np.linspace(*plane[1][:2], 100)
        y = np.linspace(*plane[1][2:], 100)
        X, Y = np.meshgrid(x, y)
        Z = plane[1][0] + plane[1][1] * X + plane[1][2] * Y
        dist_gradient = (dist_array.get(key, 0) - min_dist) / (max_dist - min_dist)
        color = plt.cm.PiYG(dist_gradient)
        ax.contourf(X, Y, Z, alpha=0.90, colors=[color])
        print(f'Roof {i} drawn')

    ax.set_aspect('equal', 'box')
    ax.set_ylim(*[limit + (-0.01 if i == 0 else 0.01) for i, limit in enumerate(ax.get_ylim())])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.PiYG, norm=plt.Normalize(vmin=min_dist, vmax=max_dist))
    fig.colorbar(sm, ax=ax, label='Średni błąd [m]')
    ax.set_title('Średni błąd dla powierzchni dachów w modelu 3D [m]')
    ax.set_xlabel('Y [m]')
    ax.set_ylabel('X [m]')
    plt.show()
            


if __name__ == '__main__':
    # czytanie pliku laz - należy podać ścieżkę do pliku
    cloud_df, coords = lazReadWrite_building(r'C:\pythonProject\semestr 4\stand_i_konw_3D\zaj4\78776_1433581_M-34-47-D-c-1-3-4-3.laz')
    print('laz done')
    # czytanie pliku gml - należy podać ścieżkę do folderu z plikami gml (program sam znajduje pliki gml pasujące do chmury punktów)
    roofPoints_map = gmlReadWrite(r'C:\pythonProject\semestr 4\stand_i_konw_3D\zaj4\Modele_3D', coords)
    print('gml done')
    # obliczanie płaszczyzn dachów
    planes_map = roofPointPlanes(roofPoints_map)
    print('planes done')
    # obliczanie odległości punktów chmury od płaszczyzn dachów
    distance_arr = distance(planes_map, cloud_df)
    print('distance done')
    # rysowanie płaszczyzn dachów
    draw_planes(planes_map, distance_arr)