import os
import cv2
import glob
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from pdf2image import convert_from_path

def plot_geo(blocktype, color, name, jpg_file):
  la_imag = cv2.imread(jpg_file)
  L, A, D = la_imag.shape

  for geo in blocktype['Geometry']:
    bbox = geo['BoundingBox']
    x = round(bbox['Left']*A)
    y = round(bbox['Top']*L)
    w = round(bbox['Width']*A)
    h = round(bbox['Height']*L)
    cv2.rectangle(la_imag, (x, y), (x+w, y+h), color, 2)

  cv2.imwrite('/home/'+name+'.jpg', la_imag)
  return la_imag

def plot_points(la_imag, blocktype, color):
  L, A, D = la_imag.shape
  coor = []
  inte = []
  cb = 0
  for geo in blocktype['Geometry']:
    cb = cb + 1
    bbox = geo['BoundingBox']
    x = round(bbox['Left']*A)
    y = round(bbox['Top']*L)
    w = round(bbox['Width']*A)
    h = round(bbox['Height']*L)
    density = round(w/h)
    if density == 0 or density == 1:
      density = 2
    for p in range(density):
      cv2.circle(la_imag,(x,y),3,color)
      cv2.circle(la_imag,(x,y+h),3,color)
      if density:
        cv2.circle(la_imag,(x,y+round(h/2)),3,(50,50,50))
        inte.append([x,y+round(h/2)])
      coor.append([x,y,cb])
      coor.append([x,y+h,cb])
      x = x + round(w/density)
      
    x = round(bbox['Left']*A)
    last_p = [[x,y+round(h/2),cb],[x+w,y,cb],[x+w,y+round(h/2),cb],[x+w,y+h,cb]]
    for lp in last_p:
      cv2.circle(la_imag,(lp[0],lp[1]),3,color)
      coor.append(lp)
  
  cv2.imwrite('/home/points.jpg', la_imag)
  return coor, inte

def plot_Delaunay(la_imag, points, tri):
  edges = []
  for t in tri:
    v1 = t[0]; coor_1 = points[v1]
    v2 = t[1]; coor_2 = points[v2]
    v3 = t[2]; coor_3 = points[v3]
    if [v1,v2] not in edges and [v2,v1] not in edges:
      edges.append([v1,v2])
    if [v1,v3] not in edges and [v3,v1] not in edges:
      edges.append([v1,v3])
    if [v2,v3] not in edges and [v3,v2] not in edges:
      edges.append([v2,v3])
    cv2.line(la_imag, (coor_1[0], coor_1[1]), (coor_2[0], coor_2[1]), (0,0,255), 2)
    cv2.line(la_imag, (coor_1[0], coor_1[1]), (coor_3[0], coor_3[1]), (0,0,255), 2)
    cv2.line(la_imag, (coor_2[0], coor_2[1]), (coor_3[0], coor_3[1]), (0,0,255), 2)
    
  cv2.imwrite('/home/delaunay.jpg', la_imag)
  return edges

def plot_beta_skeleton(edges, points, la_imag):
  print('Bordes iniciales:',len(edges))
  edges_b = []
  removed_edges = []
  for edge in edges:
    [v1,v2] = [edge[0],edge[1]]
    [coor_1,coor_2] = [points[v1],points[v2]]
    center = [(coor_1[0]+coor_2[0])/2,(coor_1[1]+coor_2[1])/2]

    diam = ((coor_1[0]-coor_2[0])**2 + (coor_1[1]-coor_2[1])**2)**0.5
    if len(coor_1)==3 and len(coor_2)==3:
      if coor_1[2] == coor_2[2]:
        removed_edges.append(edge)
    
    for point in points:
      dist =  ((center[0]-point[0])**2 + (center[1]-point[1])**2)**0.5
      if dist < diam/2:
        if edge in edges:
          removed_edges.append(edge)
          break
  
  n = 0
  for edge in edges:
    if edge not in removed_edges:
      n = n+1
      edges_b.append(edge)
      [v1,v2] = [edge[0],edge[1]]
      [coor_1,coor_2] = [points[v1],points[v2]]
      cv2.line(la_imag, (coor_1[0], coor_1[1]), (coor_2[0], coor_2[1]), (0,0,255), 2)

  print('Bordes beta-skeleton:',n)
  cv2.imwrite('/home/beta-skeleton.jpg', la_imag)
  return edges_b

def graph_representation(edges_b,coor):
  boxes_pair = []
  min_edges = []
  graph_edges = []
  for ed in edges_b:
    [coor1,coor2] = [coor[ed[0]], coor[ed[1]]]
    [box1,box2] = [coor1[2], coor2[2]]
    edge_dist = ((coor1[0]-coor2[0])**2 + (coor1[1]-coor2[1])**2)**0.5
    if [box1, box2] not in boxes_pair and [box2, box1] not in boxes_pair:
      boxes_pair.append([box1, box2])
      min_edges.append(edge_dist)
      graph_edges.append(ed)
    else:
      if [box1, box2] in boxes_pair:
        ind = boxes_pair.index([box1, box2])
        if edge_dist < min_edges[ind]:
          boxes_pair[ind] = [box1, box2]
          min_edges[ind] = edge_dist
          graph_edges[ind] = ed
      else:
        ind = boxes_pair.index([box2, box1])
        if edge_dist < min_edges[ind]:
          boxes_pair[ind] = [box1, box2]
          min_edges[ind] = edge_dist
          graph_edges[ind] = ed
  return graph_edges

def plot_graph_rep(la_imag, coor, graph_edges, name, n):
  for ge in graph_edges:
    [coor_1,coor_2] = [coor[ge[0]], coor[ge[1]]]
    cv2.line(la_imag, (coor_1[0], coor_1[1]), (coor_2[0], coor_2[1]), (0,0,255), 2)
  cv2.imwrite('/home/'+n+'_graph_representation_'+name+'.jpg', la_imag)

def graph_blocktype(blocktype, name, jpg_file, json_file, n):
  el_json = pd.read_json(json_file)
  block = el_json[el_json['BlockType']==blocktype]
  print('número de',blocktype,':',len(block))

  la_imag = plot_geo(block, (44,155,12),name,jpg_file)
  coor, inte = plot_points(la_imag, block, (255,0,0))
  points = np.array([c[0:2] for c in coor])
  tri_Dela = Delaunay(points)
  edges = plot_Delaunay(la_imag, points, tri_Dela.simplices)

  la_imag = plot_geo(block, (44,155,12),name,jpg_file)
  all_points = coor+inte
  edges_b = plot_beta_skeleton(edges, all_points, la_imag)

  la_imag = plot_geo(block, (44,155,12),name,jpg_file)
  graph_edges = graph_representation(edges_b,coor)
  plot_graph_rep(la_imag, coor, graph_edges, name, n)

  return graph_edges

file_paths = glob.glob('proformas/**.pdf')
graphs_w = []
graphs_l = []
for path in file_paths:
  imag = convert_from_path(path)
  jpg_file = path[:-3]+'jpg'
  json_file = path[:-3]+'json'
  imag[0].save(jpg_file)
  n = os.path.basename(path); ind = n.index('_')
  n = n[0:ind]; print(n)
  #graph_edges_w = graph_blocktype('WORD', 'words', jpg_file, json_file, n)
  #graphs_w.append(graph_edges_w)
  #graph_edges_l = graph_blocktype('LINE', 'lines', jpg_file, json_file, n)
  #graphs_l.append(graph_edges_l)