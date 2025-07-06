phone_case_width = 10;
phone_case_height = 20;

inner_case_width = 9;
inner_case_height = 19;

module phone_case() {
    rotate(0, 90, 0)
    difference() {
        hull() {
            for (x = [-phone_case_width,phone_case_width], 
                 y = [-phone_case_height, phone_case_height]
            ) {
                translate([x,y,0])
                cylinder(h=6, r=5);
            }
        }
        
        hull() {
            for (x = [-inner_case_width,inner_case_width], 
                 y = [-inner_case_height, inner_case_height]
            ) {
                translate([x,y, 1])
                cylinder(h=8, r=5);
            }
        }
        
        
        // to je luknja za kamero
        translate([-6, 15, -4])
        hull() {
            for (x = [4,-4], 
                 y = [-4,4]
            ) {
                translate([x,y, -2])
                cylinder(h=16, r=2, $fn=60);
            }
        }        
    }
    
    
    translate([-6, 15, 1])
    
        difference() {
        
        hull() {
            for (x = [5,-5], 
                 y = [-5,5]
            ) {
                translate([x,y, -2])
                cylinder(h=1, r=2, $fn=60);
            }
        }  
    
            hull() {
                for (x = [4,-4], 
                     y = [-4,4]
                ) {
                    translate([x,y, -5])
                    cylinder(h=8, r=2, $fn=60);
                }
            }      
            
        }
}

