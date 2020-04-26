// Gmsh project created on Tue Apr 21 14:57:20 2020
//+
Point(1) = {-1, -1, 0, 1.0};
//+
Point(2) = {-1, 1, 0, 1.0};
//+
Point(3) = {1, 1, 0, 1.0};
//+
Point(4) = {1, -1, 0, 1.0};
//+
Point(5) = {1, 0, 0, 1.0};

//+
Line(1) = {3, 5};
//+
Line(2) = {5, 4};
//+
Line(3) = {4, 1};
//+
Line(4) = {1, 2};
//+
Line(5) = {2, 3};
//+
Curve Loop(1) = {5, 1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Physical Surface("domain") = {1};
//+
Physical Curve("boundary") = {5, 1, 2, 3, 4};
