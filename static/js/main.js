// Function to open the modal
function openModal() {
    document.getElementById("myModal").style.display = "block";
}

function openModalChoose() {
    document.getElementById("chooseModal").style.display = "block";
}


// Function to close the modal
function closeModal() {
    document.getElementById("myModal").style.display = "none";
}

// Function to close the modal
function closeModalChoose() {
    document.getElementById("chooseModal").style.display = "none";
}

// Function to set the selected option
function setSelectedOption(option) {
    document.getElementById('selected_option').value = option;
    // Update the selected option text in the box
    document.getElementById('selected_option_text').innerText = option;
    // Close the modal after selecting an option
    closeModal();
}

// Function to submit the form
function submitForm() {
    // Submit the form asynchronously using JavaScript Fetch API
    fetch('/analyze', {
        method: 'POST',
        body: new FormData(document.getElementById('analysisForm'))
    })
        .then(response => {
            if (response.ok) {
                // Open the chooseModal modal when the form submission is successful
                openModalChoose();
            } else {
                // Handle errors if the form submission fails
                console.error('Form submission failed');
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

function submitForm_Brand() {
    // Submit the form asynchronously using JavaScript Fetch API
    fetch('/brand_detail', {
        method: 'POST',
        body: new FormData(document.getElementById('analysisForm'))
    })
        .then(response => {
            if (response.ok) {
                // Open the chooseModal modal when the form submission is successful
                openModalChoose();
            } else {
                // Handle errors if the form submission fails
                console.error('Form submission failed');
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

function submitForm_compare() {
    // Submit the form asynchronously using JavaScript Fetch API
    fetch('/brand_compare', {
        method: 'POST',
        body: new FormData(document.getElementById('analysisForm'))
    })
        .then(response => {
            if (response.ok) {
                // Open the chooseModal modal when the form submission is successful
                openModalChoose();
            } else {
                // Handle errors if the form submission fails
                console.error('Form submission failed');
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

function redirectToPlot() {
    window.location.href = "plot";
}

function redirectToTable() {
    window.location.href = "table";
}

// Add click event listener to the entire document
document.addEventListener("click", function () {
    // Call the redirect function when the document is clicked
    redirectToMenu();
    redirectToTable();
});

