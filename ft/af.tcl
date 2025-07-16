
# Analyze property files 
clear -all 

set FT_PATH /mnt/c/Users/huijie/Desktop/AssertionForge/ft
set CLOCK clock
set RESET reset

# Initialize coverage for both stimuli models and COI 
check_cov -init -model all -type all 

analyze -clear 

analyze -sv12 \
    ${FT_PATH}/baud_gen.sv \
    ${FT_PATH}/uart_parser.sv \
    ${FT_PATH}/uart_rx.sv \
    ${FT_PATH}/uart_top.sv \
    ${FT_PATH}/uart_tx.sv \
    ${FT_PATH}/uart2bus_top.sv \
    ${FT_PATH}/uart_sva.sv

elaborate -top uart2bus_top

clock ${CLOCK}
reset ${RESET}

# Define clock and reset signals and run proof


# Get design information to check general complexity
get_design_info

# Run proof on all assertions with a time limit 
set_max_trace_length 10
prove -all

# Get proof results
set proofs_status [get_status [get_property_list -include {type {assert} disabled {0}}]]

# Output the proof results
puts "proofs: $proofs_status"

# Check if any properties failed (have status 'cex' or 'falsified')
set failed_props [get_property_list -include {type {assert} status {cex falsified}}]

if {[llength $failed_props] > 0} {
    puts "WARNING: Some properties failed with counterexample:"
    foreach prop $failed_props {
        puts "  - $prop"
    }
    puts "Continuing with coverage calculation despite property failures..."
}

# Measure coverage for both stimuli models and COI regardless of property failures
check_cov -measure -type all -verbose

# Coverage reporting script 
set coverage_models {functional statement toggle expression branch}
set coverage_types {stimuli coi}

puts "\nCOVERAGE REPORT"
puts "TYPE|MODEL|COVERAGE"
puts "--------------------"

foreach type $coverage_types {
    foreach model $coverage_models {
        if {$type == "coi"} {
            set coverage_data [check_cov -report -model $model -type checker -checker_mode coi]
        } else {
            set coverage_data [check_cov -report -model $model -type $type]
        }
        if {[regexp {([0-9.]+)%} $coverage_data match coverage]} {
            puts "$type|$model|$coverage"
        } else {
            puts "$type|$model|N/A"
        }
    }
}

puts "### COVERAGE_REPORT_START ###"    
set undetectable_coverage [check_cov -list -status undetectable -checker_mode coi]
puts "### UNDETECTABLE_START ###"
puts $undetectable_coverage
puts "### UNDETECTABLE_END ###"

set unprocessed_coverage [check_cov -list -status unprocessed -checker_mode coi]
puts "### UNPROCESSED_START ###"
puts $unprocessed_coverage
puts "### UNPROCESSED_END ###"

puts "### COVERAGE_REPORT_END ###"
