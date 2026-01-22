// Registration
document.getElementById("registerForm")?.addEventListener("submit", function(e) {
    e.preventDefault();

    const username = document.getElementById("regUsername").value;
    const password = document.getElementById("regPassword").value;

    localStorage.setItem(username, password);
    alert("Registration successful!");
    window.location.href = "index.html";
});

// Login
document.getElementById("loginForm")?.addEventListener("submit", function(e) {
    e.preventDefault();

    const username = document.getElementById("loginUsername").value;
    const password = document.getElementById("loginPassword").value;

    const storedPassword = localStorage.getItem(username);

    if (storedPassword === password) {
        alert("Login successful!");
        // redirect to dashboard
        // window.location.href = "dashboard.html";
    } else {
        alert("Invalid username or password!");
    }
});
