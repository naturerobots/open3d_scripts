#!python3
import open3d as o3d
import argparse
import numpy as np
import cv2




def main():

  parser = argparse.ArgumentParser(
      description='Reconstruction of point cloud data (PCD) to a reconstructed mesh using Open3D',
  )
  parser.add_argument('-i', '--input', dest='input', required=True, action='store',
                      help='point cloud input ply file', type=str)
  parser.add_argument('-o', '--output', dest='output', required=True, action='store',
                      help='mesh output ply file', type=str)
  parser.add_argument('-v', '--voxel-size', dest='voxel_size', required=False, default=0.08, action='store',
                      help='down sample voxel-size', type=float)
  parser.add_argument('-n', '--normals', dest='normals', action='store_true', required=False, default=False,
                      help='esitamte normals')
  parser.add_argument('--n-radius', dest='n_radius', action='store', required=False, default=0.3, type=float,
                      help='radius to consider for the normal estimation.')
  parser.add_argument('--n-max-nn', dest='n_max_nn', action='store', required=False, default=30, type=float,
                      help='maximum number of nearest neighbors for normal estimation.')
  

  args = parser.parse_args()
  
  cloud = o3d.io.read_point_cloud(args.input)
  print("downsample cloud with voxel_size", args.voxel_size)
  cloud = cloud.voxel_down_sample(voxel_size=args.voxel_size)
  if args.normals or not cloud.has_normals():
      if not cloud.has_normals():
          print("No normals in the point cloud, estimating normals...")
      else:
          print("Estimate normals...")
      cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
          radius=args.n_radius, max_nn=args.n_max_nn))
      
  # remove point if normal decribes a horizontal plane
  for point_id in range(len(cloud.points)):
    normal = cloud.normals[point_id]
    if abs(normal[2]) > 0.5:
      #remove point
      cloud.points[point_id] = [0,0,0]
      cloud.normals[point_id] = [0,0,0]


  #make colors of cloud points black
  cloud.colors = o3d.utility.Vector3dVector(np.zeros((len(cloud.points), 3)))
  
  vis = o3d.visualization.Visualizer()
  vis.create_window()
  vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
  vis.get_render_option().point_size = 2.0
  vis.add_geometry(cloud)
  vis.capture_screen_image("file.jpg", do_render=True)
  vis.destroy_window()






  # make cloud 2d by removing z axis
  # for point_id in range(len(cloud.points)):
  #   point = cloud.points[point_id]
  #   cloud.points[point_id] = [point[0], point[1], 0]
  #   normal = cloud.normals[point_id]
  #   cloud.normals[point_id] = [normal[0], normal[1], 0]


  # # Konvertiere die Punktwolke in ein Voxelgitter
  # voxel_size = 0.01  # Voxelgröße in Metern
  # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud, voxel_size)

  # # Extrahiere das Tiefenbild aus dem Voxelgitter
  # depth_image = voxel_grid.extract_depth_image()

  # # Erstelle ein Farbbild mit einheitlicher Farbe (optional)
  # width, height = depth_image.shape
  # color_image = np.ones((height, width, 3), dtype=np.uint8) * 255

  # # Konvertiere das Bild in ein OpenCV-Bildformat
  # image_cv = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

  # #detect lines with hough transform in image
  # lines = cv2.HoughLines(depth_image, 1, np.pi/180, 100)

  # #draw lines in image
  # image_with_lines = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)
  # for line in lines:
  #   rho, theta = line[0]
  #   a = np.cos(theta)
  #   b = np.sin(theta)
  #   x0 = a*rho
  #   y0 = b*rho
  #   x1 = int(x0 + 1000*(-b))
  #   y1 = int(y0 + 1000*(a))
  #   x2 = int(x0 - 1000*(-b))
  #   y2 = int(y0 - 1000*(a))
  #   cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    


  # cv2.imshow("Linien", image_with_lines)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  # o3d.io.write_point_cloud(args.output, cloud)

if __name__ == "__main__":
  main()