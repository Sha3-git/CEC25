import DragDropUploader from "../components/DragDropUploader"
import { TextField, Box, Button } from "@mui/material";
import { ThemeProvider, createTheme } from "@mui/material/styles";
const theme = createTheme({
    palette: {
        mode: "dark", // Dark mode
    },
});

export default function Predict() {
    return (
        <>
            <div className="container mt-5" style={{ height: "100%" }}>
                <div className="row">
                    <div className="col-lg-6">
                        <div className="mt-5 mb-3">
                            <DragDropUploader />

                        </div>
                        <Button
                            variant="contained"
                            color="primary"
                            sx={{
                                marginBottom: 2,
                                backgroundColor: "primary.main",
                                color: "text.primary",
                                '&:hover': {
                                    backgroundColor: "primary.dark",
                                },
                            }}
                        >
                            Submit
                        </Button>
                    </div>
                    <div className="col-lg-6"></div>
                </div>

            </div>
        </>
    )
}
