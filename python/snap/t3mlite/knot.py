from snappy.snap.t3mlite.mcomplex import *
from sage.all import Rational, vector, Matrix
from snappy.snap.t3mlite.simplex import *
import random
class BarycentricPoint(object):
    def __init__(self, c0, c1, c2, c3):
        c0 = Rational(c0)
        c1 = Rational(c1)
        c2 = Rational(c2)
        c3 = Rational(c3)
        self.vector = vector([c0,c1,c2,c3])

        if sum(self.vector) != 1:
            raise Exception("Barycentric point doesn't sum to 1")

    def __repr__(self):
        return self.vector.__repr__()

    def __eq__(self, other):
        return self.vector == other.vector

    def has_negative_coordinate(self):
        for l in self.vector:
            if l<0:
                return True
        return False

    def negative_coordinates(self):
        return [l for l in self.vector if l<0]
    
    def zero_coordinates(self):
        return [i for i in range(4) if self.vector[i] == 0]

    def is_on_boundary_face(self):
        return len(self.zero_coordinates()) == 1
    
    def convex_combination(self, other, t):
        c0,c1,c2,c3 = self.vector*(1-t)+other.vector*t
        return BarycentricPoint(c0,c1,c2,c3)

    def transform(self, matrix):
        v = self.vector
        new_v = matrix*v
        return BarycentricPoint(*new_v)

    def to_3d_point(self):
        return self.vector[0:3]

    def permute(self, perm):
        """
        Start with a permutation perm, which represents a map from the vertices of
        a tetrahedron T in which a point lies, to the vertices of another 
        tetrahedron S which is glued to T. Then, translate
        the barycentric coordinates of the point in T to the corresponding
        barycentric coordinates of the point in S.
        On the interior, this doesn't make much sense; it really should be 
        used for points on the common boundary triangle.
        """
        v = self.vector
        new_v = [0]*4
        for i in range(4):
            new_v[perm[i]] = v[i]
        return BarycentricPoint(*new_v)
    
P = BarycentricPoint

class BarycentricArc(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def reversed(self):
        return BarycentricArc(self.end, self.start)
    
    def trim_start(self):
        ts = []
        if not self.start.has_negative_coordinate():
            return self
        for i in range(4):
            start_i, end_i = self.start.vector[i], self.end.vector[i]
            if start_i == end_i:
                continue
            else:
                t = -start_i/(end_i-start_i)
                if 0 <= t <= 1:
                    ts.append(t)
                else:
                    continue

        for t in sorted(ts): #find the closest point to start which has nonnegative coordinates along the line from start to end
            new_start = self.start.convex_combination(self.end,t)
            if new_start.has_negative_coordinate():
                continue
            else:
                return BarycentricArc(new_start, self.end)

        return None


    def trim_end(self):
        return self.reversed().trim_start().reversed()
        
    def __repr__(self):
        return '[{},{}]'.format(self.start, self.end)

    def __eq__(self, other):
        return (self.start == other.start) and (self.end == other.end)

    def direction(self):
        return self.end.vector-self.start.vector

    def transform(self, matrix):
        new_start = self.start.transform(matrix)
        new_end = self.end.transform(matrix)
        return BarycentricArc(new_start, new_end)

    def is_point(self):
        return self.start == self.end

    def to_3d_points(self):
        return (self.start.to_3d_point(), self.end.to_3d_point())



Arc = BarycentricArc

A = vector([Rational('1'),Rational('0'),Rational('0'),Rational('1')])
B = vector([Rational('-1'),Rational('1'),Rational('0'),Rational('1')])
C =vector([Rational('-1'),Rational('-1'),Rational('0'),Rational('1')])
N =vector([Rational('0'),Rational('0'),Rational('1'),Rational('1')])
S = vector([Rational('0'),Rational('0'),Rational('-1'),Rational('1')])


ABNS = Matrix([A,B,N,S]).T
BCNS = Matrix([B,C,N,S]).T
CANS = Matrix([C,A,N,S]).T

three = [ABNS, BCNS, CANS]


NABC = Matrix([N,A,B,C]).T
ABCS = Matrix([A,B,C,S]).T

two = [NABC, ABCS]

ABNSI = ABNS.inverse()
BCNSI = BCNS.inverse()
CANSI = CANS.inverse()

three_inv = [ABNSI, BCNSI, CANSI]

NABCI = NABC.inverse()
ABCSI = ABCS.inverse()

two_inv = [NABCI, ABCSI]

#trans_32_1 = [M*N for M in three for N in two_inv]
#trans_23_1 = [M*N for M in two for N in three_inv]
trans_32 = [N*M for M in three for N in two_inv]
trans_23 = [N*M for M in two for N in three_inv]


one_half = Rational('1/2')
one_third = Rational('1/3')
one_sixth = Rational('1/6')
c1 = Rational('1/5')
c2 = Rational('1/7')
c3 = Rational('23/35')
t3m_face_to_midpoint = {7:BarycentricPoint(one_third,one_third,one_third,0),
                        11:BarycentricPoint(one_third,one_third,0,one_third),
                        13:BarycentricPoint(one_third,0,one_third,one_third),
                        14:BarycentricPoint(0,one_third,one_third,one_third)}
t3m_face_to_point = {7:BarycentricPoint(c1,c2,c3,0),
                        11:BarycentricPoint(c1,c2,0,c3),
                        13:BarycentricPoint(c1,0,c2,c3),
                        14:BarycentricPoint(0,c1,c2,c3)}

def plot_arcs(arcs):
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from itertools import combinations
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    tet_points = [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
    for p1, p2 in combinations(tet_points,2):
        x = [p1[0],p2[0]]
        y = [p1[1],p2[1]]
        z = [p1[2],p2[2]]        
        ax.plot(x,y,z,color='black')
    for arc in arcs:
        p1, p2 = arc.to_3d_points()
        x = [p1[0],p2[0]]
        y = [p1[1],p2[1]]
        z = [p1[2],p2[2]]
        ax.plot(x,y,z)
    plt.show()

def smash(mc):    
    if len(mc.Vertices) > 1:
        for e in mc.Edges:
            if mc.smash_star(e):
                return 1
    return 0

def smash_all(mc):
    while len(mc.Vertices)>1:
        ans = smash(mc)

def test():
    M = snappy.Triangulation('K3a1(1,0)')
    MC = snappy.snap.t3mlite.Mcomplex(M._unsimplified_filled_triangulation())
    smash_all(MC)
    MC._add_arcs_around_valence_one_edge()
    for i in range(1):
        MC._two_three_simplify(3,20)
        print([tet.arcs for tet in MC.Tetrahedra if tet.arcs])
        print('tets:{}'.format(len(MC.Tetrahedra)))
    most_arcs = max(MC.Tetrahedra, key=lambda tet: len(tet.arcs)).arcs
    plot_arcs(most_arcs)

def test_simplification():
    M = snappy.Triangulation('K3a1(1,0)')
    MC = snappy.snap.t3mlite.Mcomplex(M._unsimplified_filled_triangulation())
    smash_all(MC)
    for i in range(100):
        if len(MC.Tetrahedra)>1:            
            MC._two_three_simplify(10,5)
            MC.rebuild()
            print('tets:{}'.format(len(MC.Tetrahedra)))
        else:
            break


def test_simplification_randomize():
    M = snappy.Triangulation('K3a1(1,0)')
    MC = snappy.snap.t3mlite.Mcomplex(M._unsimplified_filled_triangulation())
    smash_all(MC)

    for i in range(10):
        if len(MC.Tetrahedra)>1:            
            MC.randomize()
            print(MC.Vertices)
            print('tets:{}'.format(len(MC.Tetrahedra)))
        else:
            break


class BarycentricTetrahedronEmbedding(object):
    def __init__(self, tetrahedron, vertex_images):
        self.tetrahedron = tetrahedron
        assert len(vertex_images)==4
        for vi in vertex_images:
            assert len(vi)==3
            for i in range(3):
                vi[i] = Rational(vi[i])
            vi.append(Rational('1'))
        self.matrix = Matrix(vertex_images).T
        assert self.matrix.det() > 0
        
    def transfer_arcs_from(self, other_embedding):
        transition_matrix = self.matrix.inverse()*other_embedding.matrix

        return [arc.transform(transition_matrix) for arc in other_embedding.tetrahedron.arcs]

    def info(self):
        self.tetrahedron.info()
        print(self.matrix)


def barycentric_face_embedding(arrow):
    """
    arrow goes from north pole to vertex a
    """
    arrow = arrow.copy()
    next_arrow = arrow.glued()
    vertex_images = [None]*4
    vertex_images[VertexIndex[arrow.tail()]] = [0,0,1] #N
    vertex_images[VertexIndex[arrow.head()]] = [1,0,0] #A
    arrow.opposite()
    vertex_images[VertexIndex[arrow.tail()]] = [-1,-1,0] #C
    vertex_images[VertexIndex[arrow.head()]] = [-1,1,0] #B

    next_vertex_images = [None]*4
    next_vertex_images[VertexIndex[next_arrow.tail()]] = [1,0,0] #A
    next_vertex_images[VertexIndex[next_arrow.head()]] = [0,0,-1] #S
    next_arrow.opposite()
    next_vertex_images[VertexIndex[next_arrow.tail()]] = [-1,-1,0] #C
    next_vertex_images[VertexIndex[next_arrow.head()]] = [-1,1,0] #B

    return [BarycentricTetrahedronEmbedding(arrow.Tetrahedron,vertex_images),
            BarycentricTetrahedronEmbedding(next_arrow.Tetrahedron, next_vertex_images)]


def barycentric_edge_embedding(arrow):
    """
    arrow goes from a to b
    """
    assert len(arrow.linking_cycle())==3
    arrow = arrow.copy()
    next = arrow.glued()
    after_next = next.glued()

    vertex_images = [None]*4
    vertex_images[VertexIndex[arrow.tail()]] = [1,0,0] #A
    vertex_images[VertexIndex[arrow.head()]] = [-1,1,0] #B
    arrow.opposite()
    vertex_images[VertexIndex[arrow.tail()]] = [0,0,-1] #S
    vertex_images[VertexIndex[arrow.head()]] = [0,0,1] #N

    next_vertex_images = [None]*4
    next_vertex_images[VertexIndex[next.tail()]] = [-1,1,0] #B
    next_vertex_images[VertexIndex[next.head()]] = [-1,-1,0] #C
    next.opposite()
    next_vertex_images[VertexIndex[next.tail()]] = [0,0,-1] #S
    next_vertex_images[VertexIndex[next.head()]] = [0,0,1] #N

    after_next_vertex_images = [None]*4
    after_next_vertex_images[VertexIndex[after_next.tail()]] = [-1,-1,0] #C
    after_next_vertex_images[VertexIndex[after_next.head()]] = [1,0,0] #A
    after_next.opposite()
    after_next_vertex_images[VertexIndex[after_next.tail()]] = [0,0,-1] #S
    after_next_vertex_images[VertexIndex[after_next.head()]] = [0,0,1] #N

    return [BarycentricTetrahedronEmbedding(arrow.Tetrahedron,vertex_images),
            BarycentricTetrahedronEmbedding(next.Tetrahedron, next_vertex_images),
            BarycentricTetrahedronEmbedding(after_next.Tetrahedron,after_next_vertex_images)]
    
    
    
class BarycentricFaceEmbedding(object):
    def __init__(self, arrow):
        next_arrow = arrow.glued()
        arrow.head()


class BarycentricEdgeEmbedding(object):
    def __init__(self, arrow):
        pass

    
    
def test_embedding_functions():
    M = snappy.Triangulation('K3a1(1,0)')
    MC = snappy.snap.t3mlite.Mcomplex(M._unsimplified_filled_triangulation())

    MC.jiggle()
    MC.rebuild()
    valence_three = [e for e in MC.Edges if e.valence()==3]
    arrow = valence_three[0].get_arrow()
    be1, be2, be3 = barycentric_edge_embedding(arrow)
    print(be1.matrix)
    print('')
    print(be2.matrix)
    print('')
    print(be3.matrix)
    print('')
    arrow.Tetrahedron.info()
    print([VertexIndex[arrow.tail()],VertexIndex[arrow.head()]])
    arrow_op = arrow.copy()
    arrow_op = arrow_op.opposite()
    print([VertexIndex[arrow_op.tail()],VertexIndex[arrow_op.head()]])

    
    next_arrow = arrow.glued()
    next_arrow.Tetrahedron.info()
    print([VertexIndex[next_arrow.tail()],VertexIndex[next_arrow.head()]])
    next_arrow_op = next_arrow.copy()
    next_arrow_op.opposite()
    print([VertexIndex[next_arrow_op.tail()],VertexIndex[next_arrow_op.head()]])


    after_next_arrow = next_arrow.glued()
    after_next_arrow.Tetrahedron.info()
    print([VertexIndex[after_next_arrow.tail()],VertexIndex[after_next_arrow.head()]])
    after_next_arrow_op = after_next_arrow.copy()
    after_next_arrow_op.opposite()
    print([VertexIndex[after_next_arrow_op.tail()],VertexIndex[after_next_arrow_op.head()]])


def search_for_best_two_to_three(MC):
    num_valence_three = MC.EdgeValences.count(3)
    best_face = None
    best_diff = 0
    for face in MC.Faces:
        corner = face.Corners[0]
        tet = corner.Tetrahedron
        copy = MC.copy()
        copy_tet = copy.Tetrahedra[tet.Index]
        copy.two_to_three(corner.Subsimplex, copy_tet)
        copy.rebuild()
        diff = copy.EdgeValences.count(3)-num_valence_three
        if diff > best_diff:
            best_face = face
            best_diff = diff

    return best_face, best_diff
        
def apply_best_two_to_three_move(MC):
    face, diff = search_for_best_two_to_three(MC)
    if face == None:
        face = random.choice(MC.Faces)

    corner = face.Corners[0]
    MC.two_to_three(corner.Subsimplex, corner.Tetrahedron)
    MC.rebuild()


def apply_many_two_to_three(MC, n):
    for i in range(n):
        for tet in MC.Tetrahedra:
            if tet.arcs:
                f = random.choice([F0,F1,F2,F3])
                MC.two_to_three(f, tet)
#                MC.rebuild()
                break


def apply_one_three_to_two(MC):
    did_simplify = 0
    for edge in MC.Edges:
        if edge.valence() == 3:
            if MC.three_to_two(edge):
                did_simplify = 1
                MC.rebuild()
                break
            
    return did_simplify    
        

def two_three_simplify(MC, num_blowups, num_attempts):
    for i in range(num_attempts):
        print('T:{},E:{},V:{}'.format( len(MC.Tetrahedra), len(MC.Edges), len(MC.Vertices) ))
        for j in range(num_blowups):
            print('going up')
            apply_best_two_to_three_move(MC)
            check_arcs_connected(MC)
        print('going_down')
        eliminate_random_valence_three(MC)
        check_arcs_connected(MC)



def eliminate_random_valence_three(MC):
    did_simplify = 0
    progress = 1
    while progress:
        progress = 0
        valence_three_edges = [edge for edge in MC.Edges if edge.valence()==3]
        random.shuffle(valence_three_edges)
        for edge in valence_three_edges:
            if MC.three_to_two(edge):
                progress, did_simplify = 1, 1
                break
        MC.rebuild()
    return did_simplify


def check_arcs_connected(MC):
    for tet in MC.Tetrahedra:
        pts = [arc.start for arc in tet.arcs]
        pts.extend([arc.end for arc in tet.arcs])        
        for pt in pts:
            is_on_face = False
            for x in pt.vector:
                if x == 0:
                    is_on_face = True
                    assert pts.count(pt) == 1
            if not is_on_face:
                assert pts.count(pt) == 2


def trace_knot(MC):
    return trace_knot_starting_at(*get_arc(MC))
                
def trace_knot_starting_at(arc, tet):
    """
    Follow the arcs around from one tetrahedron to the next.
    """
    
    start_point, start_tet = arc.start, tet
    knot = [(arc,tet)]
    next_arc, next_tet = continue_arc(arc,tet)

    while next_arc.start != start_point or next_tet != start_tet:
        print(next_arc, next_tet)
        knot.append((next_arc,next_tet))
        next_arc, next_tet = continue_arc(next_arc,next_tet)
        
    return knot


FacesByIndex = [F0,F1,F2,F3]

def continue_arc(arc, tet):

    endpoint = arc.end
    if arc.end.is_on_boundary_face(): #move to neighboring tetrahedron
        face_bitmap = FacesByIndex[arc.end.zero_coordinates()[0]]
        perm = tet.Gluing[face_bitmap]
        endpoint = endpoint.permute(perm)
        tet = tet.Neighbor[face_bitmap]
        other_arcs = tet.arcs[:] 
    else: 
        other_arcs = tet.arcs[:]
        other_arcs.remove(arc)

#    print(endpoint)
#    print(other_arcs)
#    print(tet)
    for other_arc in other_arcs:
        if other_arc.start == endpoint:
            return other_arc, tet
        elif other_arc.end == endpoint:
            return other_arc.reversed(), tet
        else:
            continue
    raise Exception("No matching arc found")

def get_arc(MC):
    for tet in MC.Tetrahedra:
        for arc in tet.arcs:
            return arc, tet


def trefoil_triangulation():
    M = snappy.Triangulation("K3a1(1,0)")
    MC = snappy.snap.t3mlite.Mcomplex(M._unsimplified_filled_triangulation())
    smash_all(MC)
    MC.rebuild()
    MC._add_arcs_around_valence_one_edge()
    return MC

def knot_triangulation(name):
    M = snappy.Triangulation(name+"(1,0)")
    MC = snappy.snap.t3mlite.Mcomplex(M._unsimplified_filled_triangulation())
    smash_all(MC)
    MC.rebuild()
    MC._add_arcs_around_valence_one_edge()
    return MC

def trefoil_triangulation_no_arcs():
    M = snappy.Triangulation("K3a1(1,0)")
    MC = snappy.snap.t3mlite.Mcomplex(M._unsimplified_filled_triangulation())
    smash_all(MC)
    MC.rebuild()
#    MC._add_arcs_around_valence_one_edge()
    return MC

