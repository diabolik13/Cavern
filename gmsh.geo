// Gmsh project created on Mon Dec  9 14:33:24 2019
//+
Point(1) = {-1, -1, 0, 1.0};
//+
Point(2) = {-1, 1, 0, 1.0};
//+
Point(3) = {1, 1, 0, 1.0};
//+
Point(4) = {1, -1, 0, 1.0};
//+
Point(5) = {-1, 0.5, 0, 1.0};
//+
Point(6) = {-1, -0.5, 0, 1.0};
//+
Point(7) = {-1, 0.3, 0, 1.0};
//+
Point(8) = {-1, -0.3, 0, 1.0};
//+
Point(9) = {-0.8, 0.3, 0, 1.0};
//+
Point(10) = {-0.8, -0.3, 0, 1.0};
//+
Circle(1) = {5, 7, 9};
//+
Circle(2) = {10, 6, 8};
//+
Circle(3) = {10, 8, 6};
//+
Line(4) = {6, 1};
//+
Line(5) = {1, 4};
//+
Line(6) = {4, 3};
//+
Line(7) = {3, 2};
//+
Line(8) = {2, 5};
//+
Line(9) = {9, 10};
//+
Recursive Delete {
  Curve{2}; 
}
//+
Curve Loop(1) = {7, 8, 1, 9, 3, 4, 5, 6};
//+
Plane Surface(1) = {1};
//+
Physical Curve("cave") = {1, 9, 3};
//+
Physical Curve("top") = {7};
//+
Physical Curve("right") = {6};
//+
Physical Curve("left") = {8, 4};
//+
Physical Curve("bottom") = {5};
//+
Physical Surface("interior") = {1};
