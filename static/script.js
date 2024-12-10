// Preview the image after it's uploaded
function previewImage() {
    const uploadImage = document.getElementById('uploadImage');
    const imagePreview = document.getElementById('imagePreview');
    const submitButton = document.getElementById('submitButton');
    const deleteButton = document.getElementById('deleteButton');
    const saveButton = document.getElementById('saveButton');
    const editButton = document.getElementById('editButton');

    if (uploadImage.files && uploadImage.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.innerHTML = `<img src="${e.target.result}" alt="Uploaded Image">`;
        };
        reader.readAsDataURL(uploadImage.files[0]);

        // Show the submit and delete buttons once an image is selected
        submitButton.style.display = 'block';
        deleteButton.style.display = 'block';
        saveButton.style.display = 'block';
        editButton.style.display = 'block';
    }
}

// Delete the image and reset the form
function deleteImage() {
    const uploadImage = document.getElementById('uploadImage');
    const imagePreview = document.getElementById('imagePreview');
    const submitButton = document.getElementById('submitButton');
    const deleteButton = document.getElementById('deleteButton');
    const saveButton = document.getElementById('saveButton');
    const editButton = document.getElementById('editButton');

    const extractedImageContainer = document.getElementById('extractedImageContainer');

    // Reset the file input
    uploadImage.value = '';
    
    // Clear the image preview
    imagePreview.innerHTML = '';

    // Clear the extracted text
    extractedImageContainer.querySelectorAll('.grid-item').forEach(item => {
        item.innerHTML = '';
    });

    // Hide the submit, delete buttons, and extracted data container
    submitButton.style.display = 'none';
    deleteButton.style.display = 'none';
    saveButton.style.display = 'none';
    editButton.style.display = 'none';
    extractedImageContainer.style.display = 'none';
}

// Extract data from the uploaded image using OCR
function extractImage() {
    const uploadImage = document.getElementById('uploadImage').files[0];

    if (!uploadImage) {
        alert('Please select an image to extract.');
        return;
    }

    const formData = new FormData();
    formData.append('image', uploadImage);

    fetch('/extract', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Check if there's an error
        if (data.error) {
            alert(data.error);
            return;
        }

        // Populate form fields with extracted data
        document.getElementById('email').value = data.emails.length ? data.emails.join(', ') : 'N/A';
        document.getElementById('phone').value = data.phoneNumbers.length ? data.phoneNumbers.join(', ') : 'N/A';
        document.getElementById('designation').value = data.roles.length ? data.roles.join(', ') : 'N/A';
        document.getElementById('pincode').value = data.pinCodes.length ? data.pinCodes.join(', ') : 'N/A';
        document.getElementById('name').value = data.name.length ? data.name.join(', ') : 'N/A';
        document.getElementById('company').value = data.company.length ? data.company.join(', ') : 'N/A';
        document.getElementById('address').value = data.address.length ? data.address.join(', ') : 'N/A';
        // Show the form container
        document.getElementById('extractedImageContainer').style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('There was an error processing the image.');
    });
}

// Make grid items editable
function editGrid() {
    const formElements = document.querySelectorAll('#extractedDataForm input');
    formElements.forEach(input => {
        input.disabled = false;  // Enable editing
    });
}

// Save the edited content and export to Excel
function saveGrid() {
    const formData = {
        name: document.getElementById('name').value,
        designation: document.getElementById('designation').value,
        company: document.getElementById('company').value,
        phone: document.getElementById('phone').value,
        email: document.getElementById('email').value,
        // website: document.getElementById('website').value,
        pincode: document.getElementById('pincode').value,
        address: document.getElementById('address').value

    };
    console.log(formData)
    // Disable the form inputs after saving
    const formElements = document.querySelectorAll('#extractedDataForm input');
    formElements.forEach(input => {
        input.disabled = true;  // Disable editing
    });
    // Export data to Excel
    exportToExcel(formData);
}

// Function to export the data to Excel
function exportToExcel(data) {
    const ws = XLSX.utils.json_to_sheet([data]);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Extracted Data");

    // Download Excel file
    XLSX.writeFile(wb, "ExtractedData.xlsx");
}

