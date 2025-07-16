set FT_PATH /mnt/c/Users/huijie/Desktop/AssertionForge/ft
set CLOCK clock
set RESET reset

check_cov -init -model all -type all 

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

get_design_info
set_max_trace_length 100
prove -all

# Get proof results for enabled assertions
# Use get_properties -type assert to get assertion objects, then filter them
set enabled_assertions [get_properties -type assert -filter {disabled == 0}]
set proofs_status [get_status $enabled_assertions]

# Output the proof results
puts "proofs: $proofs_status"

# Check if any properties failed (have status 'cex' or 'falsified')
# Use get_properties -type assert with a filter for status
set failed_props [get_properties -type assert -filter {status == "cex" || status == "falsified"}]

if {[llength $failed_props] > 0} {
    puts "WARNING: Some properties failed with counterexample:"
    foreach prop $failed_props {
        # For each failed property object, print its name
        puts "  - [get_name $prop]"
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

# Report proof results
# report 
# check_cov -measure -type {coi proof}
# check_cov -report -report_file cov.rpt -force
# report -summary -force -result -file design.fpv.rpt