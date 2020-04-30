// Gmsh project created on Mon Apr 27 00:24:42 2020
//+
Point(1) = {1.15, 1.15, 0, 2.0};
//+
Point(2) = {1.15, -1.15, 0, 2.0};
//+
Point(3) = {-1.15, -1.15, 0, 2.0};
//+
Point(4) = {-1.15, 1.15, 0, 2.0};
//+
Point(5) = {-1.15, 0.15, 0, 0.1};
//+
Point(6) = {-1.15, -0.15, 0, 0.1};
//+
Point(7) = {-1.1, 0.1, 0, 0.1};
//+
Point(8) = {-1.1, -0.1, 0, 0.1};
//+
Point(9) = {-1.15, -0.1, 0, 0.1};
//+
Point(10) = {-1.15, 0.1, 0, 0.1};
//+
Line(1) = {5, 4};
//+
Line(2) = {4, 1};
//+
Line(3) = {1, 2};
//+
Line(4) = {2, 3};
//+
Line(5) = {3, 6};
//+
Line(6) = {8, 7};
//+
Circle(7) = {5, 10, 7};
//+
Circle(8) = {8, 9, 6};
//+
Curve Loop(1) = {1, 2, 3, 4, 5, -8, 6, -7};
//+
Plane Surface(1) = {1};
//+
Physical Curve("cavern", 11) = {7, 6, 8};
//+
Physical Surface("domain", 1) = {1};