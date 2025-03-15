import React, { useState } from "react";
import { Box, Typography, Snackbar, Alert } from "@mui/material";
import { useDropzone } from "react-dropzone";

export default function DragDropUploader() {
  const [image, setImage] = useState(null);
  const [error, setError] = useState("");
  const [openSnackbar, setOpenSnackbar] = useState(false);

  const onDrop = (acceptedFiles) => {
    const file = acceptedFiles[0];

    // Check if the file is a PNG image
    if (file && file.type === "image/png") {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result); // Set the image preview
        setError(""); // Clear any previous errors
        setOpenSnackbar(true); // Show success message
      };
      reader.readAsDataURL(file);
    } else {
      setError("Only PNG images are allowed!");
      setOpenSnackbar(true); // Show error message
    }
  };

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: ".png", // Only accept PNG files
  });

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "#262b2e",
        color: "white",
        height: "100%",
        padding: 2,
        paddingBottom: 5,
        borderRadius: '15px'
      }}
    >
      <Typography variant="h5" gutterBottom>
        Drag & Drop Image Upload (PNG Only)
      </Typography>
      <Box
        {...getRootProps()}
        sx={{
          border: "2px dashed #ddd",
          borderRadius: "8px",
          padding: "40px",
          textAlign: "center",
          backgroundColor: "#0f1214",
          width: "80%",
          maxWidth: "500px",
          cursor: "pointer",
        }}
      >
        <input {...getInputProps()} />
        <Typography variant="body1">Drag your PNG image here</Typography>
        <Typography variant="body2" color="textSecondary">
          Or click to select one
        </Typography>
      </Box>
      {image && (
        <Box mt={2}>
          <img src={image} alt="Uploaded Preview" style={{ maxWidth: "100%" }} />
        </Box>
      )}

      <Snackbar
        open={openSnackbar}
        autoHideDuration={3000}
        onClose={() => setOpenSnackbar(false)}
      >
        <Alert severity={error ? "error" : "success"} onClose={() => setOpenSnackbar(false)}>
          {error || "Image uploaded successfully!"}
        </Alert>
      </Snackbar>
    </Box>
  );
}
