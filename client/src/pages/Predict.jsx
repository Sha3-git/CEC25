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
                Hello
                <DragDropUploader />
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
        </>
    )
}
