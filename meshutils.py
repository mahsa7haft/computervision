import scipy
import numpy as np

def writeply(X,color,tri,filename):
    """
    Save out a triangulated mesh to a ply file
    
    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        vertex coordinates shape (3,Nvert)
        
    color : 2D numpy.array (dtype=float)
        vertex colors shape (3,Nvert)
        should be float in range (0..1)
        
    tri : 2D numpy.array (dtype=float)
        triangular faces shape (Ntri,3)
        
    filename : string
        filename to save to    
    """
    f = open(filename,"w");
    f.write('ply\n');
    f.write('format ascii 1.0\n');
    f.write('element vertex %i\n' % X.shape[1]);
    f.write('property float x\n');
    f.write('property float y\n');
    f.write('property float z\n');
    f.write('property uchar red\n');
    f.write('property uchar green\n');
    f.write('property uchar blue\n');
    f.write('element face %d\n' % tri.shape[0]);
    f.write('property list uchar int vertex_indices\n');
    f.write('end_header\n');

    C = (255*color).astype('uint8')
    
    for i in range(X.shape[1]):
        f.write('%f %f %f %i %i %i\n' % (X[0,i],X[1,i],X[2,i],C[0,i],C[1,i],C[2,i]));
    
    for t in range(tri.shape[0]):
        f.write('3 %d %d %d\n' % (tri[t,1],tri[t,0],tri[t,2]))

    f.close();


def pruning(boxlimits, trithresh, pts2L, pts2R, pts3, cpix):
    """
    prunes the pts3 and cpix with the given trithresh and boxlimits.
    
    Parameters
    ----------
    boxlimits : list of ints, threshold limits
        for box pruning
    
    trithresh : float 
        threshold value for triangle pruning
        
    pts2L,pts2R : 2D numpy.array (dtype=float) 
    
    pts3 : 2D numpy.array (dtype=float)
        vertex coordinates shape (3,Nvert)
        
    cpix : 2D numpy.array (dtype=float)
        vertex colors shape (3,Nvert)
        should be float in range (0..1)
        
    tri : 2D numpy.array (dtype=float)
        triangular faces shape (Ntri,3)
    
    Returns
    -------
    tri:  2D numpy.array (dtype=float)
        triangular faces shape (Ntri,3)
    
    pts3: 2D numpy.array (dtype=float)
        shape (3,Nvert)
        prooned
    
    cpix: 2D numpy.array (dtype=float)
        vertex colors shape (3,Nvert)
        should be float in range (0..1)
        prooned
   
    """
    a = np.argwhere(pts3[0])
    j = 0
    for i in range(3):
        a = np.argwhere(pts3[i] < boxlimits[j])
        b = np.argwhere(pts3[i] > boxlimits[j+1])
        pts3 = np.delete(pts3, np.append(a, b), axis=1)
        cpix = np.delete(cpix, np.append(a, b), axis=1)
        pts2L = np.delete(pts2L, np.append(a, b), axis=1)
        pts2R = np.delete(pts2R, np.append(a, b), axis=1)
        j+=2
    #
    # triangulate the 2D points to get the surface mesh
    #
    tris = scipy.spatial.Delaunay(pts2L.T)
    tri = tris.simplices
    a = pts2L.shape

    #
    # triangle pruning
    #
    edges_1 = np.linalg.norm(pts3.T[tri[:,0]] - pts3.T[tri[:,1]], axis=-1)
    edges_2 = np.linalg.norm(pts3.T[tri[:,1]] - pts3.T[tri[:,2]], axis=-1)
    edges_3 = np.linalg.norm(pts3.T[tri[:,0]] - pts3.T[tri[:,2]], axis=-1)
    good_triangle_mask = (edges_1<trithresh) * (edges_2<trithresh) * (edges_3<trithresh)
    
    todelete = np.argwhere(good_triangle_mask==False)
    newtri = np.delete(tri.T,todelete,axis=1)
    newtri = newtri.T
    
    tokeep = np.unique(newtri.flatten())
    allinds = np.arange(pts3.shape[1])
    out = np.delete(allinds,tokeep)
    
    pts3 = pts3[:, tokeep]
    cpix = np.delete(cpix,out, axis=1)

    map = np.zeros(a[1])
    map[tokeep]= np.arange(0,tokeep.shape[0])
    tri = map[newtri]

    return (tri,pts3, cpix)

def smoothIt(tri, pts3, n):
    """
    smoothes out the mesh
    
    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        vertex coordinates shape (3,Nvert)
        
    tri : 2D numpy.array (dtype=float)
        triangular faces shape (Ntri,3)
    
    n: number of times we want to smooth
        
    Return
    ----------
    pts3 : 2D numpy.array (dtype=float)
        vertex coordinates shape (3,Nvert)
        smoothed n times
    
    """
    neighbors = {i:set() for i in range(0,pts3.shape[1])}
    for i in tri:
        ind1 = int(i[0]); ind2 = int(i[1]); ind3 = int(i[2])
        neighbors[ind1].add(ind2)
        neighbors[ind1].add(ind3)
        neighbors[ind2].add(ind1)
        neighbors[ind2].add(ind3)
        neighbors[ind3].add(ind2)
        neighbors[ind3].add(ind1)

    smoothed = np.zeros(pts3.shape)
    
    counter = 0
    while counter < n:
        for i in range(pts3.shape[1]):
            temp = np.array(list(neighbors[i]))
            smoothed[:,i] = np.mean(pts3[:,temp], axis=1)
        counter+=1
        pts3 = smoothed
    
    return pts3
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    