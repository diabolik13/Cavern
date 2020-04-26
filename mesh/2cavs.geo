// Gmsh project created on Tue Apr 14 13:15:35 2020
//+
Point(1) = {1, 1, 0, 1.0};
//+
Point(2) = {-1, 1, 0, 1.0};
//+
Point(3) = {-1, -1, 0, 1.0};
//+
Point(4) = {1, -1, 0, 1.0};
//+
Point(5) = {-1, 0.3, 0, 0.1};
//+
Point(6) = {-0.9, 0.2, 0, 0.1};
//+
Point(7) = {-1, 0.2, 0, 0.1};
//+
Point(8) = {-0.9, -0.2, 0, 0.1};
//+
Point(9) = {-1, -0.2, 0, 0.1};
//+
Point(10) = {-1, -0.3, 0, 0.1};
//+
Point(11) = {1, 0.3, 0, 0.1};
//+
Point(12) = {1, 0.2, 0, 0.1};
//+
Point(13) = {0.9, 0.2, 0, 0.1};
//+
Point(14) = {1, -0.2, 0, 0.1};
//+
Point(15) = {1, -0.3, 0, 0.1};
//+
Point(16) = {0.9, -0.2, 0, 0.1};
//+
Line(1) = {2, 1};
//+
Line(2) = {1, 11};
//+
Line(3) = {13, 16};
//+
Line(4) = {15, 4};
//+
Line(5) = {4, 3};
//+
Line(6) = {3, 10};
//+
Line(7) = {8, 6};
//+
Line(8) = {5, 2};
//+
Circle(9) = {5, 7, 6};
//+
Circle(10) = {8, 9, 10};
//+
Circle(11) = {13, 12, 11};
//+
Circle(12) = {15, 14, 16};
//+
Curve Loop(1) = {1, 2, -11, 3, -12, 4, 5, 6, -10, 7, -9, 8};
//+
Plane Surface(1) = {1};
//+
//+
Physical Curve("cavern1", 11) = {9, 7, 10};
//+
Physical Curve("cavern2", 12) = {11, 3, 12};
//+
Physical Surface("domain", 1) = {1};