from tsgo.container.sample_polyhedron import h_box, cube, triangular_pyramid, pointy_star, ribbed_cage

h_box_model = h_box()
pointy_star_model = pointy_star()
cage = ribbed_cage()

# plot the hbox with the point (1.5, 0.5, 1.5) highlighted

# FOR ACTIVE PLANES
# if __name__ == "__main__":
#     from src.visualizer import plot_result
#     from src.point import Point
    
#     points_xyz = [[1.5, -0.5, 1.5]]
    
#     face_colors = [0]
#     active_planes = [i for i, f in enumerate(h_box_model.faces) if h_box_model._is_active_plane(f, Point(1.5, -0.5, 1.5))]

#     plot_result(points_xyz=points_xyz, D=0.05, container=h_box_model, show_normals=False, face_colors=active_planes)


# FOR FOOTPOINTS
# if __name__ == "__main__":
#     from src.visualizer import plot_result
#     from src.point import Point

#     test_point = Point(0.0, 0.0, 4.0)
#     points_xyz = [[test_point.x, test_point.y, test_point.z]]
#     footpoints = pointy_star_model.get_active_footpoints(test_point)
#     points_xyz.extend([[fp.x, fp.y, fp.z] for fp in footpoints])


#     plot_result(points_xyz=points_xyz, D=0.05, container=pointy_star_model, show_normals=False, colors=['red'] + ['blue'] * len(footpoints))


# VISUALIZE A POLYHEDRON
if __name__ == "__main__":
    from tsgo.plotting.visualizer import plot_result
    points = [[0, 0, 0]]
    D = 0.01

    plot_result(points_xyz=points, D=D, container=cage, show_normals=False)