/*
A KBase module: TaxonomyAbundance
*/

module TaxonomyAbundance {
    typedef structure {
        string report_name;
        string report_ref;
    } ReportResults;

    typedef structure {
        string amplicon_matrix_ref;
        string test_row_attri_ref;
        string attri_mapping_ref;
        float threshold;
        string taxonomy_level;
        string grouping_label;
    } TaxonomyAbundanceInput;

    /*
        This example function accepts any number of parameters and returns results in a KBaseReport
    */
    funcdef run_TaxonomyAbundance(TaxonomyAbundanceInput params) returns (ReportResults output) authentication required;

};
