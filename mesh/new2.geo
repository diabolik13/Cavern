// Gmsh project created on Mon Feb  3 23:27:10 2020
//+
Point(1) = {-1, -1, -0, 1.0};
//+
Point(2) = {-1, 1, -0, 1.0};
//+
Point(3) = {1, 1, -0, 1.0};
//+
Point(4) = {1, -1, -0, 1.0};
//+
Point(5) = {-1, 0, -0, 1.0};
//+
Point(6) = {-1, 0.4, -0, 1.0};
//+
Point(7) = {-1, -0.4, -0, 1.0};
//+
Circle(1) = {7, 5, 6};
//+
Line(2) = {7, 1};
//+
Line(3) = {1, 4};
//+
Line(4) = {4, 3};
//+
Line(5) = {3, 2};
//+
Line(6) = {2, 6};
//+
Curve Loop(1) = {1, -6, -5, -4, -3, -2};
//+
Plane Surface(1) = {1};
//+
Physical Curve("cavern", 1) = {1};
//+
Physical Curve("left") = {6, 2};
//+
Physical Curve("top") = {5};
//+
Physical Curve("right") = {4};
//+
Physical Curve("bottom") = {3};
//+
Plane Surface(2) = {1};
//+
Physical Surface("interior") = {1};
