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
        string attri_mapping_ref;
        float threshold;
        int taxonomy_level;
        mapping<string, string> grouping_label;
    } TaxonomyAbundanceInput;

    /*
        This example function accepts any number of parameters and returns results in a KBaseReport
    */
    funcdef run_TaxonomyAbundance(TaxonomyAbundanceInput params) returns (ReportResults output) authentication required;

};
