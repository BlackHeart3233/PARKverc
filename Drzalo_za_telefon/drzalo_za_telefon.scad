include <phone_holder.scad>;


module drzalo() {

    translate([-5, 17, 15])
    cube([10, 15, 2]);
    
    translate([-5, 17, 10])
    cube([10, 2, 5]);     


    translate([-5, 17, -17])
    cube([10, 15, 2]);     

    translate([-5, 17, -15])
    cube([10, 2, 5]);  
    
    
    translate([-27, 17, -6])
    cube([2, 10, 10]);     

    translate([-25, 17, -6])
    cube([3, 2, 10]);
    
    
    translate([13,43,-60])
    vakumsko_drzalo_left();
    
    translate([-17,155,-60])
    vakumsko_drzalo_right();
    
    translate([-25,25,-15])
    cube([50,10,30]);

    
}


rotate([0, 90, 90])
translate([0, 0, 18])
phone_case();

color([0.5, 0.5, 0.5])
drzalo();


module vakumsko_drzalo_left() {
// Parametri
cup_radius = 20;      // zunanji premer priseska (v mm)
cup_height = 8;       // višina kupole
lip_thickness = 1;    // debelina roba priseska
stem_radius = 2;      // premer zgornjega nastavka
stem_height = 6;      // višina zgornjega nastavka


module suction_cup_left() {
    // Glavni del priseska – kupola
    difference() {
        scale([1, 1, 0.4]) // sploščen polkrog
            sphere(r=cup_radius);

        // Izreži notranjost, da ustvarimo tanek rob (lip)
        translate([0, 0, -1])
            scale([1, 1, 0.4])
                sphere(r=cup_radius - lip_thickness);
    }

    // Zgornji nastavek
    translate([0, 0, cup_height])
        cylinder(h=stem_height, r=stem_radius, $fn=64);
}


// Prikaži model
suction_cup_left();

     // sredinski členek 1
    difference() {
        translate([-2.5, 0, 20])
        rotate([0, 90, 0])
        cylinder(h = 5, r = 8);

        // luknja v sredini
        translate([-3, 0, 20])
        rotate([0, 90, 0])
        cylinder(h = 7, r = 2);
    }
}

module vakumsko_drzalo_right() {
// Parametri
cup_radius = 20;      // zunanji premer priseska (v mm)
cup_height = 8;       // višina kupole
lip_thickness = 1;    // debelina roba priseska
stem_radius = 2;      // premer zgornjega nastavka
stem_height = 6;      // višina zgornjega nastavka


module suction_cup() {
    // Glavni del priseska – kupola
    difference() {
        scale([1, 1, 0.4]) // sploščen polkrog
            sphere(r=cup_radius);

        // Izreži notranjost, da ustvarimo tanek rob (lip)
        translate([0, 0, -1])
            scale([1, 1, 0.4])
                sphere(r=cup_radius - lip_thickness);
    }

    // Zgornji nastavek
    translate([0, 0, cup_height])
        cylinder(h=stem_height, r=stem_radius, $fn=64);
    
     difference() {
        translate([-2.5, 0, 21])
        rotate([0, 90, 0])
        cylinder(h = 5, r = 8);

        // luknja v sredini
        translate([-3, 0, 21])
        rotate([0, 90, 0])
        cylinder(h = 7, r = 2);
    }
}



// Prikaži model
suction_cup();
}



module mehanska_noga_zgoraj() {
    // sredinski členek 1
    difference() {
    translate([0, 50, 0])
    rotate([0, 90, 0])
    cylinder(h = 5, r = 10);

    // luknja v sredini (rahlo daljša)
    translate([-4, 50, 0])
    rotate([0, 90, 0])
    cylinder(h = 9.1, r = 2);
}


    difference() {
        translate([11, 50, 0])
        rotate([0, 90, 0])
        cylinder(h = 5, r = 10);

        translate([10, 50, 0])
        rotate([0, 90, 0])
        cylinder(h = 8, r = 2);
    }

    translate([0, 45, 8])  
    cube([16, 10, 6.5]);

    //šravfek
    translate([-2, 50, 0])
    color("gray")
    rotate([0, 90, 0])
    cylinder(h = 20, r = 2);

    // sredinski členek 2
    difference() {
        translate([30, 50, 0])
        rotate([0, 90, 0])
        cylinder(h = 5, r = 10);

        translate([29, 50, 0])
        rotate([0, 90, 0])
        cylinder(h = 8, r = 2);
    }

    difference() {
        translate([41, 50, 0])
        rotate([0, 90, 0])
        cylinder(h = 5, r = 10);

        translate([40, 50, 0])
        rotate([0, 90, 0])
        cylinder(h = 8, r = 2);
    }

    translate([28, 50, 0])
    color("gray")
    rotate([0, 90, 0])
    cylinder(h = 20, r = 2);
    
    
    translate([30, 45, 8])  
    cube([16, 10, 6.5]);
}

translate([-25,50,-12])
    mehanska_noga_zgoraj();


module mehanska_noga_spodaj() {

    difference() {
        translate([5.5, 50, 0])
        rotate([0, 90, 0])
        color("gray")
        cylinder(h = 5, r = 8);

        translate([5, 50, 0])  // popravljeno: isto kot zunanji valj
        rotate([0, 90, 0])
        cylinder(h = 10, r = 2);  // rahlo daljši
    }

    difference() {
        translate([35.5, 50, 0])
        rotate([0, 90, 0])
        color("gray")
        cylinder(h = 5, r = 8);

        translate([35, 50, 0])  
        rotate([0, 90, 0])
        cylinder(h = 10, r = 2);
    }

    translate([5.5, 55, 0])
rotate([-30, 0, 0])
color("gray")
cube([5,45,5]);
    
    
        translate([35.5, 44, 0])
rotate([-155, 0, 0])
color("gray")
cube([5,45,5]);
    
        difference() {
    translate([0, 105, -26])
    rotate([0, 90, 0])
    cylinder(h = 5, r = 10);

    // luknja v sredini
    translate([-1, 105, -26])
    rotate([0, 90, 0])
    cylinder(h = 9.1, r = 2);
}


    difference() {
        translate([11, 105, -26])
        rotate([0, 90, 0])
        cylinder(h = 5, r = 10);

        translate([10, 105, -26])
        rotate([0, 90, 0])
        cylinder(h = 8, r = 2);
    }

    translate([0, 45, 8])  
    cube([16, 10, 6.5]);
    
        translate([-2, 105, -26])
    color("gray")
    rotate([0, 90, 0])
    cylinder(h = 20, r = 2);

translate([0, 95, -27])  
    rotate([60, 0, 0])
    cube([16, 10, 6.5]);
        
    
    difference() {
        translate([30, -7, -28])
        rotate([0, 90, 0])
        cylinder(h = 5, r = 10);

        translate([29, -7, -28])
        rotate([0, 90, 0])
        cylinder(h = 8, r = 2);
    }
    
    difference() {
        translate([41, -7, -28])
        rotate([0, 90, 0])
        cylinder(h = 5, r = 10);

        translate([40, -7, -28])
        rotate([0, 90, 0])
        cylinder(h = 8, r = 2);
    }

        translate([30, -3, -20])
    rotate([-65,0,0])
    cube([16, 10, 6.5]);
    
        translate([28, -7, -28])
    color("gray")
    rotate([0, 90, 0])
    cylinder(h = 20, r = 2);

    
}

translate([-25,50,-12])
    mehanska_noga_spodaj();



module ramenski_clenek_right() {
    // Okrogla 3/4 jamica
    difference() {
        translate([0, 0, 0])
        sphere(r = 8);

        // Odrežemo samo zgornjo četrtino – ostane 3/4 krogle
        translate([-20, -20, 3])  // reži višje kot za polkrog
        cube([40, 40, 20]);

        // Izrežemo notranji prostor – luknja za kroglo
        translate([0, 0, 0])
        sphere(r = 7.1);  
        // malo večja od notranje krogle
    }

    // Krogla, ki gre notri
    translate([0, 0, 0])
    sphere(r = 6.6);

    // roka ki izhaja iz krogle
    translate([0, 0, 10])
    cube([3, 3, 12], center=true);
    
    translate([0, 0, 35])
    color("gray")
    cube([5, 5, 50], center=true);
}


module ramenski_clenek_left() {
    // Okrogla 3/4 jamica
    difference() {
        translate([0, 0, 0])
        sphere(r = 8);

        // Odrežemo samo zgornjo četrtino – ostane 3/4 krogle
        translate([-20, -20, 3])  // reži višje kot za polkrog
        cube([40, 40, 20]);

        // Izrežemo notranji prostor – luknja za kroglo
        translate([0, 0, 0])
        sphere(r = 7.1);  
        // malo večja od notranje krogle
    }

    // Krogla, ki gre notri
    translate([0, 0, 0])
    sphere(r = 6.6);

    // "roka", ki izhaja iz krogle
    translate([0, 0, 10])
    cube([3, 3, 12], center=true);
    
    translate([0, 0, 35])
    color("gray")
    cube([5, 5, 50], center=true);
}

translate([-15,35,0])
rotate([90,0,180])
ramenski_clenek_right();

translate([15,35,0])
rotate([90,0,180])
ramenski_clenek_left();
