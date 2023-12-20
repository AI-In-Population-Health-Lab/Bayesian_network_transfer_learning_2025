public class test_start {
    public static void main(String[] args) throws Exception {

        // run example-------------------------------
        runnerBNTL runnerBNTL = new runnerBNTL();
//        String [] res = {"/Users/johnsong/downloads/smile_license/License.java",
//                "/Users/johnsong/Documents/Ye_lab/BNTL_source_code/000influenza_ac_topazac_slc_topazslc",
//                "folder10_Influenza_CFS_acmodel_0810_topazac_size50_slcmodel_0810_topazslc.bif_targetDataSize_1000-IG",
//                "unadjust","off","off"};

        String[] res = {"/Users/johnsong/downloads/smile_license/License.java",
                "/Users/johnsong/documents/test_bntl_jar/no_KL",
                "FLU-SLC2AC-0809-transfer20090118-IG001-NB",
                "unadjust","off","off"
        };
        runnerBNTL.main(res);

    }
}
