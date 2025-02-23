
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def update_limits(ax, all_points, margin=1):
    """Update axis limits to include all points with a given margin."""
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    ax.set_xlim(min_vals[0] - margin, max_vals[0] + margin)
    ax.set_ylim(min_vals[1] - margin, max_vals[1] + margin)
    ax.set_zlim(min_vals[2] - margin, max_vals[2] + margin)

def main():
    # Read origin and direction from stdin.
    try:
        origin = np.array(list(map(float, input("Enter origin point (x y z): ").strip().split())))
        direction = np.array(list(map(float, input("Enter direction point (x y z): ").strip().split())))
    except ValueError:
        print("Invalid input for origin or direction. Please enter 3 space-separated numbers.")
        return
    
    # Initialize list of all points for dynamic axis limits
    all_points = np.array([origin, direction])
    
    # Create figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot origin and direction points
    ax.scatter(origin[0], origin[1], origin[2], color='red', s=100, label='Origin (O)')
    ax.scatter(direction[0], direction[1], direction[2], color='green', s=100, label='Direction (D)')
    ax.text(origin[0], origin[1], origin[2], ' O', color='red', fontsize=12)
    ax.text(direction[0], direction[1], direction[2], ' D', color='green', fontsize=12)
    
    update_limits(ax, all_points)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Turn on interactive mode so we can update the plot continuously
    plt.ion()
    plt.show()
    
    while True:
        response = input("Do you want to add a new triangle? (y/n): ").strip().lower()
        if response != 'y':
            break
        
        triangle_points = []
        print("Enter 3 vertices for the triangle (each as 'x y z'):")
        for i in range(3):
            try:
                point = list(map(float, input(f"Vertex {i+1}: ").strip().split()))
                if len(point) != 3:
                    print("Please enter exactly 3 numbers.")
                    break
                triangle_points.append(point)
            except ValueError:
                print("Invalid input. Please enter 3 space-separated numbers.")
                break
        
        if len(triangle_points) != 3:
            print("Triangle not added due to invalid input. Try again.")
            continue
        
        triangle_points = np.array(triangle_points)
        
        # Create and add the triangle polygon
        verts = [triangle_points]
        triangle_poly = Poly3DCollection(verts, alpha=0.5, facecolor='blue')
        ax.add_collection3d(triangle_poly)
        
        # Plot the triangle vertices
        ax.scatter(triangle_points[:, 0], triangle_points[:, 1], triangle_points[:, 2],
                   color='blue', s=50)
        
        # Update the list of all points and adjust axis limits
        all_points = np.vstack((all_points, triangle_points))
        update_limits(ax, all_points)
        
        # Redraw the plot to show the new triangle
        plt.draw()
        plt.pause(0.1)
    
    # Turn off interactive mode and show final plot
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()

