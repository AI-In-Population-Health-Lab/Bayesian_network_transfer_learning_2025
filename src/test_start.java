public class test_start {
    public static void main(String[] args) throws Exception {

//        System.setProperty("java.library.path", "/Users/johnsong/Documents/Ye_lab/BNTL_source_code/lib");
//        System.getProperty("java.library.path");
        runnerBNTL runnerBNTL = new runnerBNTL();

        // pass into string
        String [] res = {"/Users/johnsong/Documents/Ye_lab/BNTL_source_code/000influenza_ac_topazac_slc_topazslc/",
                "folder10_Influenza_CFS_acmodel_0810_topazac_size50_slcmodel_0810_topazslc.bif_targetDataSize_1000-IG",
                "off",
                "unadjust","no","off"};

        runnerBNTL.main(res);


    }
}
