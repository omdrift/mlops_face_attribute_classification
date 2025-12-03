// JavaScript for face attribute search interface

// Configuration
const IMAGES_PER_PAGE = 20;
let currentPage = 1;
let allResults = [];

// Attribute labels mapping
const ATTRIBUTE_LABELS = {
   barbe: { 0: "Non", 1: "Oui" },
   moustache: { 0: "Non", 1: "Oui" },
   lunettes: { 0: "Non", 1: "Oui" },
   taille_cheveux: { 0: "Chauve", 1: "Court", 2: "Long" },
   couleur_cheveux: { 0: "Blond", 1: "Châtain", 2: "Roux", 3: "Brun", 4: "Gris/Blanc" }
};

// DOM elements
const searchBtn = document.getElementById('searchBtn');
const resetBtn = document.getElementById('resetBtn');
const resultsGrid = document.getElementById('resultsGrid');
const resultsCount = document.getElementById('resultsCount');
const loading = document.getElementById('loading');
const pagination = document.getElementById('pagination');
const prevPageBtn = document.getElementById('prevPage');
const nextPageBtn = document.getElementById('nextPage');
const pageInfo = document.getElementById('pageInfo');
const uploadBtn = document.getElementById('uploadBtn');
const imageUpload = document.getElementById('imageUpload');
const uploadResult = document.getElementById('uploadResult');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
   // Event listeners
   searchBtn.addEventListener('click', performSearch);
   resetBtn.addEventListener('click', resetFilters);
   uploadBtn.addEventListener('click', () => imageUpload.click());
   imageUpload.addEventListener('change', handleImageUpload);
   prevPageBtn.addEventListener('click', () => changePage(-1));
   nextPageBtn.addEventListener('click', () => changePage(1));
   
   // Load initial results (all images)
   performSearch();
});

// Get selected filters
function getSelectedFilters() {
   const filters = {
       barbe: [],
       moustache: [],
       lunettes: [],
       taille_cheveux: [],
       couleur_cheveux: []
   };
   
   // Collect all checked checkboxes
   document.querySelectorAll('input[type="checkbox"]:checked').forEach(checkbox => {
       const name = checkbox.name;
       const value = parseInt(checkbox.value);
       if (filters[name]) {
           filters[name].push(value);
       }
   });
   
   return filters;
}

// Perform search
async function performSearch() {
   const filters = getSelectedFilters();
   
   // Show loading
   loading.style.display = 'block';
   resultsGrid.innerHTML = '';
   pagination.style.display = 'none';
   
   try {
       const response = await fetch('/api/search', {
           method: 'POST',
           headers: {
               'Content-Type': 'application/json',
           },
           body: JSON.stringify(filters)
       });
       
       if (!response.ok) {
           throw new Error('Search failed');
       }
       
       const data = await response.json();
       allResults = data.images;
       currentPage = 1;
       
       // Update results count
       resultsCount.textContent = `${data.total} image${data.total !== 1 ? 's' : ''} trouvée${data.total !== 1 ? 's' : ''}`;
       
       // Display results
       displayResults();
       
   } catch (error) {
       console.error('Search error:', error);
       resultsGrid.innerHTML = '<p style="text-align: center; color: #dc3545;">Erreur lors de la recherche. Veuillez réessayer.</p>';
   } finally {
       loading.style.display = 'none';
   }
}

// Display results for current page
function displayResults() {
   resultsGrid.innerHTML = '';
   
   if (allResults.length === 0) {
       resultsGrid.innerHTML = '<p style="text-align: center; color: #6c757d; grid-column: 1 / -1;">Aucune image ne correspond aux critères sélectionnés.</p>';
       return;
   }
   
   // Calculate pagination
   const startIndex = (currentPage - 1) * IMAGES_PER_PAGE;
   const endIndex = Math.min(startIndex + IMAGES_PER_PAGE, allResults.length);
   const pageResults = allResults.slice(startIndex, endIndex);
   
   // Display images
   pageResults.forEach(image => {
       const card = createImageCard(image);
       resultsGrid.appendChild(card);
   });
   
   // Update pagination
   updatePagination();
}

// Create image card element
// ...existing code...

// Create image card element
function createImageCard(image) {
    const card = document.createElement('div');
    card.className = 'image-card';
    
    // Fix: use 'predictions' instead of 'attributes'
    const attrs = image.predictions || image.attributes || {};
    
    // Get human-readable labels
    const labels = {
        barbe: ATTRIBUTE_LABELS.barbe[attrs.barbe] || 'N/A',
        moustache: ATTRIBUTE_LABELS.moustache[attrs.moustache] || 'N/A',
        lunettes: ATTRIBUTE_LABELS.lunettes[attrs.lunettes] || 'N/A',
        taille: ATTRIBUTE_LABELS.taille_cheveux[attrs.taille_cheveux] || 'N/A',
        couleur: ATTRIBUTE_LABELS.couleur_cheveux[attrs.couleur_cheveux] || 'N/A'
    };
    
    card.innerHTML = `
        <img src="${image.path}" alt="${image.filename}" loading="lazy">
        <div class="image-card-info">
            <div class="image-card-filename" title="${image.filename}">${image.filename}</div>
            <div class="image-card-attributes">
                Barbe: ${labels.barbe}<br>
                Moustache: ${labels.moustache}<br>
                Lunettes: ${labels.lunettes}<br>
                Cheveux: ${labels.taille}, ${labels.couleur}
            </div>
        </div>
    `;
    
    // Click to view full image
    card.addEventListener('click', () => {
        window.open(image.path, '_blank');
    });
    
    return card;
}

// ...existing code...
// Update pagination controls
// ...existing code...

// Update pagination controls
function updatePagination() {
    const totalPages = Math.ceil(allResults.length / IMAGES_PER_PAGE);
    
    if (totalPages <= 1) {
        pagination.style.display = 'none';
        return;
    }
    
    pagination.style.display = 'flex';
    
    // Build pagination HTML
    let paginationHTML = `
        <button id="prevPage" class="btn" ${currentPage === 1 ? 'disabled' : ''}>← Précédent</button>
        <div class="page-numbers">
    `;
    
    // Generate page numbers with ellipsis
    const pages = generatePageNumbers(currentPage, totalPages);
    
    pages.forEach(page => {
        if (page === '...') {
            paginationHTML += `<span class="ellipsis">...</span>`;
        } else {
            const isActive = page === currentPage;
            paginationHTML += `<button class="page-btn ${isActive ? 'active' : ''}" data-page="${page}">${page}</button>`;
        }
    });
    
    paginationHTML += `
        </div>
        <div class="page-jump">
            <input type="number" id="pageInput" min="1" max="${totalPages}" value="${currentPage}" />
            <span>/ ${totalPages}</span>
            <button id="goToPage" class="btn btn-sm">Aller</button>
        </div>
        <button id="nextPage" class="btn" ${currentPage === totalPages ? 'disabled' : ''}>Suivant →</button>
    `;
    
    pagination.innerHTML = paginationHTML;
    
    // Add event listeners
    document.getElementById('prevPage').addEventListener('click', () => changePage(-1));
    document.getElementById('nextPage').addEventListener('click', () => changePage(1));
    
    // Page number buttons
    document.querySelectorAll('.page-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            goToPage(parseInt(btn.dataset.page));
        });
    });
    
    // Go to page input
    document.getElementById('goToPage').addEventListener('click', () => {
        const pageInput = document.getElementById('pageInput');
        const page = parseInt(pageInput.value);
        if (page >= 1 && page <= totalPages) {
            goToPage(page);
        }
    });
    
    // Enter key on input
    document.getElementById('pageInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const page = parseInt(e.target.value);
            const totalPages = Math.ceil(allResults.length / IMAGES_PER_PAGE);
            if (page >= 1 && page <= totalPages) {
                goToPage(page);
            }
        }
    });
}

// Generate page numbers with ellipsis
function generatePageNumbers(current, total) {
    const pages = [];
    const delta = 2; // Number of pages to show around current page
    
    // Always show first page
    pages.push(1);
    
    // Calculate range around current page
    const rangeStart = Math.max(2, current - delta);
    const rangeEnd = Math.min(total - 1, current + delta);
    
    // Add ellipsis after first page if needed
    if (rangeStart > 2) {
        pages.push('...');
    }
    
    // Add pages in range
    for (let i = rangeStart; i <= rangeEnd; i++) {
        pages.push(i);
    }
    
    // Add ellipsis before last page if needed
    if (rangeEnd < total - 1) {
        pages.push('...');
    }
    
    // Always show last page (if more than 1 page)
    if (total > 1) {
        pages.push(total);
    }
    
    return pages;
}

// Go to specific page
function goToPage(page) {
    const totalPages = Math.ceil(allResults.length / IMAGES_PER_PAGE);
    if (page >= 1 && page <= totalPages) {
        currentPage = page;
        displayResults();
        document.querySelector('.results-section').scrollIntoView({ behavior: 'smooth' });
    }
}

// Change page
function changePage(direction) {
    const totalPages = Math.ceil(allResults.length / IMAGES_PER_PAGE);
    const newPage = currentPage + direction;
    
    if (newPage >= 1 && newPage <= totalPages) {
        goToPage(newPage);
    }
}

// ...existing code...

// Change page
function changePage(direction) {
   const totalPages = Math.ceil(allResults.length / IMAGES_PER_PAGE);
   const newPage = currentPage + direction;
   
   if (newPage >= 1 && newPage <= totalPages) {
       currentPage = newPage;
       displayResults();
       // Scroll to top of results
       document.querySelector('.results-section').scrollIntoView({ behavior: 'smooth' });
   }
}

// Reset filters
function resetFilters() {
   // Uncheck all checkboxes
   document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
       checkbox.checked = false;
   });
   
   // Perform new search (will show all images)
   performSearch();
}

// Handle image upload
async function handleImageUpload(event) {
   const file = event.target.files[0];
   if (!file) return;
   
   uploadResult.innerHTML = '<div style="text-align: center;"><div class="spinner"></div><p>Analyse en cours...</p></div>';
   
   const formData = new FormData();
   formData.append('file', file);
   
   try {
       const response = await fetch('/api/predict', {
           method: 'POST',
           body: formData
       });
       
       if (!response.ok) {
           throw new Error('Prediction failed');
       }
       
       const data = await response.json();
       
       // Display results
       displayPredictionResult(file, data);
       
   } catch (error) {
       console.error('Upload error:', error);
       uploadResult.innerHTML = '<p style="text-align: center; color: #dc3545;">Erreur lors de l\'analyse. Veuillez réessayer.</p>';
   }
}

// Display prediction result
function displayPredictionResult(file, data) {
   const imageUrl = URL.createObjectURL(file);
   
   uploadResult.innerHTML = `
       <div class="upload-preview">
           <img src="${imageUrl}" alt="Uploaded image">
           <div class="upload-predictions">
               <h3>Attributs Détectés</h3>
               <div class="prediction-item">
                   <span class="prediction-label">Barbe:</span>
                   <span class="prediction-value">${data.labels.barbe}</span>
               </div>
               <div class="prediction-item">
                   <span class="prediction-label">Moustache:</span>
                   <span class="prediction-value">${data.labels.moustache}</span>
               </div>
               <div class="prediction-item">
                   <span class="prediction-label">Lunettes:</span>
                   <span class="prediction-value">${data.labels.lunettes}</span>
               </div>
               <div class="prediction-item">
                   <span class="prediction-label">Taille Cheveux:</span>
                   <span class="prediction-value">${data.labels.taille_cheveux}</span>
               </div>
               <div class="prediction-item">
                   <span class="prediction-label">Couleur Cheveux:</span>
                   <span class="prediction-value">${data.labels.couleur_cheveux}</span>
               </div>
           </div>
       </div>
   `;
}